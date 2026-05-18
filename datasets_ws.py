import os
import torch
import faiss
import faiss.contrib.torch_utils
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from os.path import join
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader
import h5py
import time

base_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]
)

base_translation_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)


def path_to_pil_img(path):
    return Image.open(path).convert("RGB")


def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (images,
        triplets_local_indexes, triplets_global_indexes).
        triplets_local_indexes are the indexes referring to each triplet within images.
        triplets_global_indexes are the global indexes of each image.
    Args:
        batch: list of tuple (images, triplets_local_indexes, triplets_global_indexes).
            considering each query to have 10 negatives (negs_num_per_query=10):
            - images: torch tensor of shape (12, 3, h, w).
            - triplets_local_indexes: torch tensor of shape (10, 3).
            - triplets_global_indexes: torch tensor of shape (12).
    Returns:
        images: torch tensor of shape (batch_size*12, 3, h, w).
        triplets_local_indexes: torch tensor of shape (batch_size*10, 3).
        triplets_global_indexes: torch tensor of shape (batch_size, 12).
    """
    images = torch.cat([e[0] for e in batch])
    triplets_local_indexes = torch.cat([e[1][None] for e in batch])
    triplets_global_indexes = torch.cat([e[2][None] for e in batch])
    for i, (local_indexes, global_indexes) in enumerate(
        zip(triplets_local_indexes, triplets_global_indexes)
    ):
        local_indexes += (
            len(global_indexes) * i
        )
    return images, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes


def _filter_indexed_sequence(sequence, indexes_to_remove):
    """Return sequence without the positions listed in indexes_to_remove.

    This helper keeps ragged sequences as Python lists. Newer NumPy versions do
    not allow np.delete() to implicitly convert lists of variable-length arrays
    into homogeneous ndarrays.
    """
    remove = set(np.asarray(indexes_to_remove, dtype=np.int64).tolist())
    return [item for index, item in enumerate(sequence) if index not in remove]


def _decode_h5_name(name):
    if isinstance(name, bytes):
        return name.decode("UTF-8")
    return str(name)


def _has_indexed_h5_layout(h5_file):
    return "image_name" in h5_file and "image_data" in h5_file


def _get_h5_image_names(h5_file):
    if _has_indexed_h5_layout(h5_file):
        return [_decode_h5_name(image_name) for image_name in h5_file["image_name"]]
    return sorted(h5_file.keys())


def _get_utm_label(path):
    parts = path.split("@")
    return (parts[1], parts[2])


def _build_exact_positives_per_query(queries_paths, database_paths):
    label_to_indexes = {}
    for index, db_path in enumerate(database_paths):
        label_to_indexes.setdefault(_get_utm_label(db_path), []).append(index)
    return [
        np.asarray(label_to_indexes.get(_get_utm_label(q_path), []), dtype=np.int32)
        for q_path in queries_paths
    ]


def _as_1d_int_array(indexes):
    if isinstance(indexes, torch.Tensor):
        indexes = indexes.cpu().numpy()
    indexes = np.asarray(indexes, dtype=np.int64)
    if indexes.ndim == 0:
        indexes = indexes.reshape(1)
    return indexes


def _flatten_unique_indexes(indexes):
    flattened = []
    for index in indexes:
        flattened.extend(_as_1d_int_array(index).tolist())
    return list(np.unique(flattened).astype(np.int32))


def _to_int_index(index):
    if isinstance(index, torch.Tensor):
        return int(index.item())
    if isinstance(index, np.ndarray):
        return int(index.item())
    return int(index)


class BaseDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache)."""

    def __init__(
        self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train", loading_queries=True
    ):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.split = split

        self.resize = args.resize
        self.test_method = args.test_method

        self.database_folder_h5_path = join(
            datasets_folder, dataset_name, split + "_database.h5"
        )
        if loading_queries:
            self.queries_folder_h5_path = join(
                datasets_folder, dataset_name, split + "_queries.h5"
            )
        else:
            self.queries_folder_h5_path = join(
                datasets_folder, dataset_name, split + "_database.h5"
            )
        database_folder_h5_df = h5py.File(self.database_folder_h5_path, "r")
        queries_folder_h5_df = h5py.File(self.queries_folder_h5_path, "r")

        self.database_name_dict = {}
        self.queries_name_dict = {}

        self.database_h5_indexed = _has_indexed_h5_layout(database_folder_h5_df)
        self.queries_h5_indexed = _has_indexed_h5_layout(queries_folder_h5_df)

        for index, database_image_name in enumerate(_get_h5_image_names(database_folder_h5_df)):
            self.database_name_dict[database_image_name] = index
        for index, queries_image_name in enumerate(_get_h5_image_names(queries_folder_h5_df)):
            self.queries_name_dict[queries_image_name] = index

        self.database_paths = sorted(self.database_name_dict)
        self.queries_paths = sorted(self.queries_name_dict)

        self.database_utms = np.array(
            [(path.split("@")[1], path.split("@")[2])
             for path in self.database_paths]
        ).astype(np.float32)
        self.queries_utms = np.array(
            [(path.split("@")[1], path.split("@")[2])
             for path in self.queries_paths]
        ).astype(np.float32)

        positive_match = getattr(args, "positive_match", "auto")
        if positive_match == "auto":
            exact_match_datasets = {"DSM", "tirmss", "VEDAI", "NEU"}
            positive_match = (
                "exact"
                if dataset_name in exact_match_datasets
                or not self.database_h5_indexed
                or not self.queries_h5_indexed
                else "distance"
            )
        if positive_match not in {"distance", "exact"}:
            raise ValueError(
                "positive_match must be one of {'auto', 'distance', 'exact'}, "
                + f"but got {positive_match}"
            )
        self.positive_match = positive_match

        if self.positive_match == "exact":
            self.soft_positives_per_query = _build_exact_positives_per_query(
                self.queries_paths, self.database_paths
            )
        else:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_utms)
            self.soft_positives_per_query = knn.radius_neighbors(
                self.queries_utms,
                radius=args.val_positive_dist_threshold,
                return_distance=False,
            )

        if args.prior_location_threshold != -1:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_utms)
            self.hard_negatives_per_query = knn.radius_neighbors(
                self.queries_utms,
                radius=args.prior_location_threshold,
                return_distance=False,
            )

        self.images_paths = list(self.database_paths) +\
            list(self.queries_paths)

        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)

        self.database_folder_h5_df = None
        self.queries_folder_h5_df = None
        database_folder_h5_df.close()
        queries_folder_h5_df.close()

        identity_transform = transforms.Lambda(lambda x: x)
        query_transforms = [
            transforms.Grayscale(num_output_channels=3)
            if self.args.G_gray
            else identity_transform,
        ]
        test_random_crop = getattr(self.args, "test_random_crop", 0)
        if test_random_crop is not None and test_random_crop > 0:
            query_transforms.append(
                transforms.RandomResizedCrop(
                    size=self.resize, scale=(1 - test_random_crop, 1)
                )
            )
        test_random_rotation = getattr(self.args, "test_random_rotation", 0)
        if test_random_rotation is not None and test_random_rotation > 0:
            query_transforms.append(
                transforms.RandomRotation(degrees=test_random_rotation)
            )
        if getattr(self.args, "test_horizontal_flip", False):
            query_transforms.append(transforms.RandomHorizontalFlip())
        query_transforms.append(base_transform)
        self.query_transform = transforms.Compose(query_transforms)

    def __getitem__(self, index):
        if self.database_folder_h5_df is None:
            self.database_folder_h5_df = h5py.File(
                self.database_folder_h5_path, "r")
            self.queries_folder_h5_df = h5py.File(
                self.queries_folder_h5_path, "r")
        if self.is_index_in_queries(index):
            if self.args.G_contrast:
                img = self.query_transform(
                    transforms.functional.adjust_contrast(self._find_img_in_h5(index), contrast_factor=3))
            else:
                img = self.query_transform(
                    self._find_img_in_h5(index))
        else:
            img = self._find_img_in_h5(index)
            img = base_transform(img)

        if self.test_method == "hard_resize":
            img = transforms.functional.resize(img, self.resize)
        else:
            img = self._test_query_transform(img)
        return img, index

    def _test_query_transform(self, img):
        """Transform query image according to self.test_method."""
        C, H, W = img.shape
        if self.test_method == "single_query":
            processed_img = transforms.functional.resize(img, min(self.resize))
        elif self.test_method == "central_crop":
            scale = max(self.resize[0] / H, self.resize[1] / W)
            processed_img = torch.nn.functional.interpolate(
                img.unsqueeze(0), scale_factor=scale
            ).squeeze(0)
            processed_img = transforms.functional.center_crop(
                processed_img, self.resize
            )
            assert processed_img.shape[1:] == torch.Size(
                self.resize
            ), f"{processed_img.shape[1:]} {self.resize}"
        elif (
            self.test_method == "five_crops"
            or self.test_method == "nearest_crop"
            or self.test_method == "maj_voting"
        ):
            shorter_side = min(self.resize)
            processed_img = transforms.functional.resize(img, shorter_side)
            processed_img = torch.stack(
                transforms.functional.five_crop(processed_img, shorter_side)
            )
            assert processed_img.shape == torch.Size(
                [5, 3, shorter_side, shorter_side]
            ), f"{processed_img.shape} {torch.Size([5, 3, shorter_side, shorter_side])}"
        return processed_img

    def _find_img_in_h5(self, index, database_queries_split=None):
        index = _to_int_index(index)
        if database_queries_split is None:
            image_name = self.images_paths[index]
            database_queries_split = "database" if index < self.database_num else "queries"
        else:
            if database_queries_split == "database":
                image_name = self.database_paths[index]
            elif database_queries_split == "queries":
                image_name = self.queries_paths[index]
            else:
                raise KeyError("Dont find correct database_queries_split!")

        if database_queries_split == "database":
            if self.database_h5_indexed:
                img = Image.fromarray(
                    self.database_folder_h5_df["image_data"][
                        self.database_name_dict[image_name]
                    ]
                )
            else:
                img = Image.fromarray(self.database_folder_h5_df[image_name][:])
        elif database_queries_split == "queries":
            if self.queries_h5_indexed:
                img = Image.fromarray(
                    self.queries_folder_h5_df["image_data"][
                        self.queries_name_dict[image_name]
                    ]
                )
            else:
                img = Image.fromarray(self.queries_folder_h5_df[image_name][:])
        else:
            raise KeyError("Dont find correct database_queries_split!")

        return img

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >"

    def get_positives(self):
        return self.soft_positives_per_query

    def get_hard_negatives(self):
        return self.hard_negatives_per_query

    def __del__(self):
        try:
            if (
                hasattr(self, "database_folder_h5_df")
                and self.database_folder_h5_df is not None
            ):
                self.database_folder_h5_df.close()
                self.queries_folder_h5_df.close()
        except Exception:
            pass

    def find_black_region(self):
        queries_folder_h5_df = h5py.File(self.queries_folder_h5_path, "r")
        queries_with_black_region = []
        for index, path in enumerate(self.queries_paths):
            if self.queries_h5_indexed:
                image_index = self.queries_name_dict[path]
                image = queries_folder_h5_df["image_data"][image_index]
            else:
                image = queries_folder_h5_df[path][:]
            if np.count_nonzero(image==0):
                queries_with_black_region.append(index)
        queries_folder_h5_df.close()
        return queries_with_black_region

    def is_index_in_queries(self, index):
        if index >= self.database_num:
            return True
        else:
            return False


class PCADataset(BaseDataset):

    def __init__(
        self, args, datasets_folder="datasets", dataset_name="pitts30k"
        ):
        super().__init__(args, datasets_folder, dataset_name, split="train")

    def __getitem__(self, index):
        if self.database_folder_h5_df is None:
            self.database_folder_h5_df = h5py.File(
                self.database_folder_h5_path, "r")
            self.queries_folder_h5_df = h5py.File(
                self.queries_folder_h5_path, "r")
        img = self._find_img_in_h5(index)
        img = base_transform(img)

        if self.test_method == "hard_resize":
            img = transforms.functional.resize(img, self.resize)
        else:
            img = self._test_query_transform(img)
        return img


class TripletsDataset(BaseDataset):
    """Dataset used for training, it is used to compute the triplets
    with TripletsDataset.compute_triplets() with various mining methods.
    If is_inference == True, uses methods of the parent class BaseDataset,
    this is used for example when computing the cache, because we compute features
    of each image, not triplets.
    """

    def __init__(
        self,
        args,
        datasets_folder="datasets",
        dataset_name="pitts30k",
        split="train",
        negs_num_per_query=10,
    ):
        super().__init__(args, datasets_folder, dataset_name, split)
        self.mining = args.mining
        self.neg_samples_num = (
            args.neg_samples_num
        )
        self.negs_num_per_query = (
            negs_num_per_query
        )
        if (
            self.mining == "full"
        ):
            self.neg_cache = [
                np.empty((0,), dtype=np.int32) for _ in range(self.queries_num)
            ]
        self.is_inference = False

        identity_transform = transforms.Lambda(lambda x: x)
        self.resized_transform = transforms.Compose(
            [
                transforms.Resize(self.resize)
                if self.resize is not None
                else identity_transform,
                base_transform,
            ]
        )

        self.query_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3)
                if self.args.G_gray
                else identity_transform,
                transforms.ColorJitter(brightness=args.brightness)
                if args.brightness != None
                else identity_transform,
                transforms.ColorJitter(contrast=args.contrast)
                if args.contrast != None
                else identity_transform,
                transforms.ColorJitter(saturation=args.saturation)
                if args.saturation != None
                else identity_transform,
                transforms.ColorJitter(hue=args.hue)
                if args.hue != None
                else identity_transform,
                transforms.RandomPerspective(args.rand_perspective)
                if args.rand_perspective != None
                else identity_transform,
                transforms.RandomResizedCrop(
                    size=self.resize, scale=(1 - args.random_resized_crop, 1)
                )
                if args.random_resized_crop != None
                else identity_transform,
                transforms.RandomRotation(degrees=args.random_rotation)
                if args.random_rotation != None
                else identity_transform,
                self.resized_transform,
            ]
        )

        if self.positive_match == "exact":
            self.hard_positives_per_query = _build_exact_positives_per_query(
                self.queries_paths, self.database_paths
            )
        else:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_utms)
            self.hard_positives_per_query = list(
                knn.radius_neighbors(
                    self.queries_utms,
                    radius=args.train_positives_dist_threshold,
                    return_distance=False,
                )
            )

        queries_without_any_hard_positive = np.where(
            np.array([len(p)
                     for p in self.hard_positives_per_query], dtype=object) == 0
        )[0]
        if len(queries_without_any_hard_positive) != 0:
            logging.info(
                f"There are {len(queries_without_any_hard_positive)} queries without any positives "
                + "within the training set. They won't be considered as they're useless for training."
            )

        self.hard_positives_per_query = _filter_indexed_sequence(
            self.hard_positives_per_query, queries_without_any_hard_positive
        )
        self.soft_positives_per_query = _filter_indexed_sequence(
            self.soft_positives_per_query, queries_without_any_hard_positive
        )
        self.queries_paths = np.delete(
            self.queries_paths, queries_without_any_hard_positive
        )

        self.images_paths = list(self.database_paths) +\
            list(self.queries_paths)
        self.queries_num = len(self.queries_paths)

        if self.mining == "msls_weighted":
            notes = [p.split("@")[-2] for p in self.queries_paths]
            try:
                night_indexes = np.where(
                    np.array([n.split("_")[0] == "night" for n in notes])
                )[0]
                sideways_indexes = np.where(
                    np.array([n.split("_")[1] == "sideways" for n in notes])
                )[0]
            except IndexError:
                raise RuntimeError(
                    "You're using msls_weighted mining but this dataset "
                    + "does not have night/sideways information. Are you using Mapillary SLS?"
                )
            self.weights = np.ones(self.queries_num)
            assert (
                len(night_indexes) != 0 and len(sideways_indexes) != 0
            ), "There should be night and sideways images for msls_weighted mining, but there are none. Are you using Mapillary SLS?"
            self.weights[night_indexes] += self.queries_num /\
                len(night_indexes)
            self.weights[sideways_indexes] += self.queries_num /\
                len(sideways_indexes)
            self.weights /= self.weights.sum()
            logging.info(
                f"#sideways_indexes [{len(sideways_indexes)}/{self.queries_num}]; "
                + "#night_indexes; [{len(night_indexes)}/{self.queries_num}]"
            )

    def __getitem__(self, index):
        if self.is_inference:
            return super().__getitem__(index)

        if self.database_folder_h5_df is None:
            self.database_folder_h5_df = h5py.File(
                self.database_folder_h5_path, "r")
            self.queries_folder_h5_df = h5py.File(
                self.queries_folder_h5_path, "r")

        query_index, best_positive_index, neg_indexes = torch.split(
            self.triplets_global_indexes[index], (1, 1, self.negs_num_per_query)
        )

        if self.args.G_contrast:
            query = self.query_transform(
                transforms.functional.adjust_contrast(self._find_img_in_h5(query_index, "queries"), contrast_factor=3))
        else:
            query = self.query_transform(
                self._find_img_in_h5(query_index, "queries"))

        positive = self.resized_transform(
            self._find_img_in_h5(best_positive_index, "database")
        )
        negatives = [
            self.resized_transform(self._find_img_in_h5(i, "database"))
            for i in neg_indexes
        ]
        images = torch.stack((query, positive, *negatives), 0)
        triplets_local_indexes = torch.empty((0, 3), dtype=torch.int)
        for neg_num in range(len(neg_indexes)):
            triplets_local_indexes = torch.cat(
                (
                    triplets_local_indexes,
                    torch.tensor([0, 1, 2 + neg_num]).reshape(1, 3),
                )
            )
        return images, triplets_local_indexes, self.triplets_global_indexes[index]

    def __len__(self):
        if self.is_inference:
            return super().__len__()
        else:
            return len(self.triplets_global_indexes)

    def compute_triplets(self, args, model, model_db=None):
        self.is_inference = True
        if self.mining == "full":
            self.compute_triplets_full(args, model, model_db)
        elif self.mining == "partial" or self.mining == "msls_weighted":
            self.compute_triplets_partial(args, model, model_db)
        elif self.mining == "random":
            self.compute_triplets_random(args, model, model_db)

    @staticmethod
    def compute_cache(args, model, model_db, subset_ds, cache_shape):
        """Compute the cache containing features of images, which is used to
        find best positive and hardest negatives."""

        if args.use_faiss_gpu:
            cache = RAMEfficient2DMatrixGPU(cache_shape, dtype=torch.float32, device=args.device)
        else:
            cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32)

        if model_db is not None:
            subset_db_dl = DataLoader(
                dataset=subset_ds[0],
                num_workers=args.num_workers,
                batch_size=args.infer_batch_size,
                shuffle=False,
                pin_memory=(args.device == "cuda"),
            )
            subset_qr_dl = DataLoader(
                dataset=subset_ds[1],
                num_workers=args.num_workers,
                batch_size=args.infer_batch_size,
                shuffle=False,
                pin_memory=(args.device == "cuda"),
            )
            model = model.eval()
            model_db = model_db.eval()

            with torch.no_grad():
                for images, indexes in tqdm(subset_db_dl, ncols=100):
                    images = images.to(args.device)
                    features = model_db(images)
                    if args.use_faiss_gpu:
                        cache[indexes] = features
                    else:
                        cache[indexes.numpy()] = features.cpu().numpy()

                for images, indexes in tqdm(subset_qr_dl, ncols=100):
                    images = images.to(args.device)
                    features = model(images)
                    if args.use_faiss_gpu:
                        cache[indexes] = features
                    else:
                        cache[indexes.numpy()] = features.cpu().numpy()
        else:
            subset_dl = DataLoader(
                dataset=subset_ds,
                num_workers=args.num_workers,
                batch_size=args.infer_batch_size,
                shuffle=False,
                pin_memory=(args.device == "cuda"),
            )
            model = model.eval()

            with torch.no_grad():
                for images, indexes in tqdm(subset_dl, ncols=100):
                    images = images.to(args.device)
                    features = model(images)
                    if args.use_faiss_gpu:
                        cache[indexes] = features
                    else:
                        cache[indexes.numpy()] = features.cpu().numpy()
        return cache

    def get_query_features(self, query_index, cache):
        query_features = cache[query_index + self.database_num]
        if query_features is None:
            raise RuntimeError(
                f"For query {self.queries_paths[query_index]} "
                + f"with index {query_index} features have not been computed!\n"
                + "There might be some bug with caching"
            )
        return query_features

    def get_best_positive_index(self, args, query_index, cache, query_features):
        hard_positives = _as_1d_int_array(self.hard_positives_per_query[query_index])
        if len(hard_positives) == 1:
            return int(hard_positives[0])

        positives_features = cache[hard_positives]
        if args.use_faiss_gpu:
            faiss_index = faiss.GpuIndexFlatL2(
                self.gpu_resources[0], args.features_dim)
        else:
            faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(positives_features)

        _, best_positive_num = faiss_index.search(
            query_features.reshape(1, -1), 1)
        if args.use_faiss_gpu:
            best_positive_num = int(best_positive_num.reshape(-1)[0].item())
        else:
            best_positive_num = int(best_positive_num.reshape(-1)[0])
        best_positive_index = hard_positives[best_positive_num].item()
        return best_positive_index

    def get_hardest_negatives_indexes(self, args, cache, query_features, neg_samples):
        neg_samples = _as_1d_int_array(neg_samples)
        neg_features = cache[neg_samples]
        if args.use_faiss_gpu:
            faiss_index = faiss.GpuIndexFlatL2(self.gpu_resources[1], args.features_dim)
        else:
            faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(neg_features)

        _, neg_nums = faiss_index.search(
            query_features.reshape(1, -1), self.negs_num_per_query
        )
        if args.use_faiss_gpu:
            neg_nums = neg_nums.reshape(-1).cpu()
        else:
            neg_nums = neg_nums.reshape(-1)
        neg_indexes = neg_samples[neg_nums].astype(np.int32)
        if not hasattr(neg_indexes, "__len__"):
            neg_indexes = np.expand_dims(neg_indexes, 0)
        return neg_indexes

    def compute_triplets_random(self, args, model, model_db):
        self.triplets_global_indexes = []

        sampled_queries_indexes = np.random.choice(
            self.queries_num, args.cache_refresh_rate, replace=False
        )

        positives_indexes = _flatten_unique_indexes(
            self.hard_positives_per_query[i] for i in sampled_queries_indexes
        )

        if model_db is not None:
            subset_db_ds = Subset(
                self, positives_indexes
            )
            subset_qr_ds = Subset(
                self, list(sampled_queries_indexes + self.database_num)
            )
            cache = self.compute_cache(
                args, model, model_db, [subset_db_ds, subset_qr_ds], (len(self), args.features_dim)
            )
        else:
            subset_ds = Subset(
                self, positives_indexes +
                list(sampled_queries_indexes + self.database_num)
            )
            cache = self.compute_cache(
                args, model, model_db, subset_ds, (len(self), args.features_dim)
            )

        if args.use_faiss_gpu:
            self.gpu_resources = []
            for i in range(2):
                res = faiss.StandardGpuResources()
                res.setTempMemory(200 * 1024 * 1024)
                self.gpu_resources.append(res)

        for query_index in tqdm(sampled_queries_indexes, ncols=100):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(
                args, query_index, cache, query_features
            )

            soft_positives = _as_1d_int_array(self.soft_positives_per_query[query_index])
            neg_indexes = np.random.choice(
                self.database_num,
                size=self.negs_num_per_query + len(soft_positives),
                replace=False,
            )

            if args.prior_location_threshold == -1:
                neg_indexes = np.setdiff1d(neg_indexes, soft_positives, assume_unique=True)[
                    : self.negs_num_per_query
                ]
            else:
                neg_indexes = np.setdiff1d(neg_indexes, soft_positives, assume_unique=True)
                hard_negatives = _as_1d_int_array(self.hard_negatives_per_query[query_index])
                neg_indexes = np.intersect1d(neg_indexes, hard_negatives, assume_unique=True)[
                    : self.negs_num_per_query
                ]

            self.triplets_global_indexes.append(
                (query_index, best_positive_index, *neg_indexes)
            )

        del cache

        if args.use_faiss_gpu:
            del self.gpu_resources

        self.triplets_global_indexes = torch.tensor(
            self.triplets_global_indexes)

    def compute_triplets_full(self, args, model, model_db):
        self.triplets_global_indexes = []

        sampled_queries_indexes = np.random.choice(
            self.queries_num, args.cache_refresh_rate, replace=False
        )

        database_indexes = list(range(self.database_num))

        if model_db is not None:
            subset_db_ds = Subset(
                self, database_indexes
            )
            subset_qr_ds = Subset(
                self, list(sampled_queries_indexes + self.database_num)
            )
            cache = self.compute_cache(
                args, model, model_db, [subset_db_ds, subset_qr_ds], (len(self), args.features_dim)
            )
        else:
            subset_ds = Subset(
                self, database_indexes +
                list(sampled_queries_indexes + self.database_num)
            )
            cache = self.compute_cache(
                args, model, model_db, subset_ds, (len(self), args.features_dim)
            )

        if args.use_faiss_gpu:
            self.gpu_resources = []
            for i in range(2):
                res = faiss.StandardGpuResources()
                res.setTempMemory(200 * 1024 * 1024)
                self.gpu_resources.append(res)

        for query_index in tqdm(sampled_queries_indexes, ncols=100):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(
                args, query_index, cache, query_features
            )

            neg_indexes = np.random.choice(
                self.database_num, self.neg_samples_num, replace=False
            )

            soft_positives = _as_1d_int_array(self.soft_positives_per_query[query_index])
            neg_indexes = np.setdiff1d(
                neg_indexes, soft_positives, assume_unique=True)

            if args.prior_location_threshold != -1:
                hard_negatives = _as_1d_int_array(self.hard_negatives_per_query[query_index])
                neg_indexes = np.intersect1d(neg_indexes, hard_negatives, assume_unique=True)

            neg_indexes = np.unique(
                np.concatenate([self.neg_cache[query_index], neg_indexes])
            )

            neg_indexes = self.get_hardest_negatives_indexes(
                args, cache, query_features, neg_indexes
            )

            self.neg_cache[query_index] = neg_indexes
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))

        del cache
        if args.use_faiss_gpu:
            del self.gpu_resources

        self.triplets_global_indexes = torch.tensor(
            self.triplets_global_indexes)

    def compute_triplets_partial(self, args, model, model_db):
        self.triplets_global_indexes = []

        if self.mining == "partial":
            sampled_queries_indexes = np.random.choice(
                self.queries_num, args.cache_refresh_rate, replace=False
            )
        elif (
            self.mining == "msls_weighted"
        ):
            sampled_queries_indexes = np.random.choice(
                self.queries_num, args.cache_refresh_rate, replace=False, p=self.weights
            )

        sampled_database_indexes = np.random.choice(
            self.database_num, self.neg_samples_num, replace=False
        )

        positives_indexes = _flatten_unique_indexes(
            self.hard_positives_per_query[i] for i in sampled_queries_indexes
        )

        database_indexes = list(sampled_database_indexes) + positives_indexes
        database_indexes = list(np.unique(database_indexes))

        if model_db is not None:
            subset_db_ds = Subset(
                self, database_indexes
            )
            subset_qr_ds = Subset(
                self, list(sampled_queries_indexes + self.database_num)
            )
            cache = self.compute_cache(
                args, model, model_db, [subset_db_ds, subset_qr_ds], (len(self), args.features_dim)
            )
        else:
            subset_ds = Subset(
                self, database_indexes +
                list(sampled_queries_indexes + self.database_num)
            )
            cache = self.compute_cache(
                args, model, model_db, subset_ds, (len(self), args.features_dim)
            )

        if args.use_faiss_gpu:
            self.gpu_resources = []
            for i in range(2):
                res = faiss.StandardGpuResources()
                res.setTempMemory(200 * 1024 * 1024)
                self.gpu_resources.append(res)

        for query_index in tqdm(sampled_queries_indexes, ncols=100):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(
                args, query_index, cache, query_features
            )

            soft_positives = _as_1d_int_array(self.soft_positives_per_query[query_index])
            neg_indexes = np.setdiff1d(
                sampled_database_indexes, soft_positives, assume_unique=True
            )

            if args.prior_location_threshold != -1:
                hard_negatives = _as_1d_int_array(self.hard_negatives_per_query[query_index])
                neg_indexes = np.intersect1d(neg_indexes, hard_negatives, assume_unique=True)

            neg_indexes = self.get_hardest_negatives_indexes(
                args, cache, query_features, neg_indexes
            )
            self.triplets_global_indexes.append(
                (query_index, best_positive_index, *neg_indexes)
            )

        del cache
        if args.use_faiss_gpu:
            del self.gpu_resources

        self.triplets_global_indexes = torch.tensor(
            self.triplets_global_indexes)


class RAMEfficient2DMatrix:
    """This class behaves similarly to a numpy.ndarray initialized
    with np.zeros(), but is implemented to save RAM when the rows
    within the 2D array are sparse. In this case it's needed because
    we don't always compute features for each image, just for few of
    them"""

    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.matrix = [None] * shape[0]

    def __len__(self):
        return len(self.matrix)

    def __setitem__(self, indexes, vals):
        assert vals.shape[1] == self.shape[1], f"{vals.shape[1]} {self.shape[1]}"
        for i, val in zip(indexes, vals):
            self.matrix[i] = val.astype(self.dtype, copy=False)

    def __getitem__(self, index):
        if hasattr(index, "__len__"):
            return np.array([self.matrix[i] for i in index])
        else:
            return self.matrix[index]


class RAMEfficient2DMatrixGPU:
    """This class behaves similarly to a numpy.ndarray initialized
    with np.zeros(), but is implemented to save RAM when the rows
    within the 2D array are sparse. In this case it's needed because
    we don't always compute features for each image, just for few of
    them"""

    def __init__(self, shape, dtype=torch.float32, device=None):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.matrix = [None] * shape[0]

    def __len__(self):
        return len(self.matrix)

    def __setitem__(self, indexes, vals):
        assert vals.shape[1] == self.shape[1], f"{vals.shape[1]} {self.shape[1]}"
        for i, val in zip(indexes, vals):
            self.matrix[i] = val.type(self.dtype).to(self.device)

    def __getitem__(self, index):
        if hasattr(index, "__len__"):
            return torch.stack([self.matrix[i] for i in index])
        else:
            return self.matrix[index]


class TranslationDataset(BaseDataset):
    """Dataset used for training, it is used to compute the pairs
    for image-to-image translation training.
    If is_inference == True, uses methods of the parent class BaseDataset.
    """

    def __init__(
        self,
        args,
        datasets_folder="datasets",
        dataset_name="pitts30k",
        split="train",
        clean_black_region=False,
        loading_queries=True
    ):
        super().__init__(args, datasets_folder, dataset_name, split, loading_queries)
        self.is_inference = False
        self.loading_queries = loading_queries

        identity_transform = transforms.Lambda(lambda x: x)
        self.resize = args.GAN_resize
        self.resized_transform = transforms.Compose(
            [
                transforms.Resize(self.resize)
                if self.resize is not None
                else identity_transform,
                base_translation_transform,
            ]
        )

        self.query_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1)
                if self.args.G_gray
                else identity_transform,
                self.resized_transform,
            ]
        )

        if self.positive_match == "exact":
            self.hard_positives_per_query = _build_exact_positives_per_query(
                self.queries_paths, self.database_paths
            )
        else:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_utms)
            self.hard_positives_per_query = list(
                knn.radius_neighbors(
                    self.queries_utms,
                    radius=0.1,
                    return_distance=False,
                )
            )

        queries_without_any_hard_positive = np.where(
            np.array([len(p)
                     for p in self.hard_positives_per_query], dtype=object) == 0
        )[0]
        if len(queries_without_any_hard_positive) != 0:
            logging.info(
                f"There are {len(queries_without_any_hard_positive)} queries without any positives "
                + "within the training set. They won't be considered as they're useless for training."
            )

        self.hard_positives_per_query = _filter_indexed_sequence(
            self.hard_positives_per_query, queries_without_any_hard_positive
        )
        self.queries_paths = np.delete(
            self.queries_paths, queries_without_any_hard_positive
        )

        if clean_black_region:
            queries_with_black_region = self.find_black_region()
            self.hard_positives_per_query = _filter_indexed_sequence(
                self.hard_positives_per_query, queries_with_black_region
            )
            self.queries_paths = np.delete(
                self.queries_paths, queries_with_black_region
            )
            if len(queries_with_black_region) != 0:
                logging.info(
                    f"There are {len(queries_with_black_region)} queries with black regions "
                    + "within the training set. They won't be considered."
                )

        self.images_paths = list(self.database_paths) +\
            list(self.queries_paths)
        self.queries_num = len(self.queries_paths)

    def __getitem__(self, index):
        if self.database_folder_h5_df is None:
            self.database_folder_h5_df = h5py.File(
                self.database_folder_h5_path, "r")
            self.queries_folder_h5_df = h5py.File(
                self.queries_folder_h5_path, "r")

        query_index, best_positive_index = torch.split(
            self.pairs_global_indexes[index], (1, 1)
        )

        if self.args.G_contrast:
            query = self.query_transform(
                transforms.functional.adjust_contrast(self._find_img_in_h5(query_index, "queries"), contrast_factor=3))
        else:
            query = self.query_transform(
                self._find_img_in_h5(query_index, "queries"))

        positive = self.resized_transform(
            self._find_img_in_h5(best_positive_index, "database")
        )
        return query, positive, self.queries_paths[query_index], self.database_paths[best_positive_index]

    def __len__(self):
        return len(self.pairs_global_indexes)

    def compute_pairs(self, args):
        self.is_inference = True
        self.compute_pairs_random(args)

    def get_best_positive_index(self, query_index):
        best_positive_index = int(_as_1d_int_array(self.hard_positives_per_query[query_index])[0])
        return best_positive_index

    def compute_pairs_random(self, args):
        self.pairs_global_indexes = []

        if self.loading_queries:
            sampled_queries_indexes = np.random.choice(
                self.queries_num, args.cache_refresh_rate, replace=False
            )
        else:
            sampled_queries_indexes = np.arange(self.queries_num)

        for query_index in sampled_queries_indexes:
            best_positive_index = self.get_best_positive_index(query_index)
            self.pairs_global_indexes.append(
                (query_index, best_positive_index)
            )

        self.pairs_global_indexes = torch.tensor(self.pairs_global_indexes)
