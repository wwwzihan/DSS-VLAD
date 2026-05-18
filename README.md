# DSS-VLAD: Supplementary Code

This repository contains the supplementary code for **DSS-VLAD: Decoupled
Spatial-Semantic VLAD Aggregation for Heterogeneous Image Geo-localization**.

## Contents

```text
.
├── train.py                  # Boson-nighttime Base Dataset training
├── train_extended.py         # generated-data augmentation training
├── eval.py                   # retrieval evaluation
├── parser.py                 # command-line options
├── datasets_ws.py            # HDF5 dataset loaders and mining
├── test.py                   # recall/localization evaluation
├── model/
│   ├── network.py            # backbone + ESP + aggregation model
│   ├── aggregation.py        # NetVLAD + DSR implementation
│   ├── cct/                  # optional CCT backbone support
│   ├── pix2pix_networks/     # third-party pix2pix components retained for compatibility
│   └── sync_batchnorm/       # synchronized batchnorm utilities
├── scripts/
├── environment.yml
├── licenses/
└── README.md
```

The DSS-VLAD implementation uses:

- Explicit Spatial Prior (ESP) before VLAD aggregation
- Dynamic Semantic Recalibration (DSR) after VLAD residual aggregation
- ResNet-18 `conv4_x` backbone and NetVLAD aggregation
- triplet loss with the baseline domain-adversarial training protocol

## Release scope

This release focuses on the DSS-VLAD implementation and the core
Boson-nighttime experiments. It includes:

- model code for DSS-VLAD;
- training and evaluation scripts for Boson-nighttime;
- pretrained checkpoints for the provided Boson-nighttime settings;
- instructions for obtaining the datasets used in the paper.

The paper also reports experiments on additional datasets and comparison
methods. Those experiments were evaluated in the same experimental pipeline
using integrated or adapted implementations, but baseline-specific adapters,
additional dataset scripts, and additional checkpoints are not included in this
release. Auxiliary scripts for parameter counting, inference-time profiling,
and visualization are also not included. Additional project code and pretrained
checkpoints will be released after publication.

## Environment

```bash
conda env create -f environment.yml
conda activate dssvlad
```

The experiments were implemented in PyTorch and run on a single NVIDIA RTX 3090
GPU. The reported Boson-nighttime runs use input resolution `512 x 512`, batch
size `4`, inference batch size `16`, Adam optimizer, learning rate `1e-4`, 100
epochs, 64 VLAD clusters, and triplet margin `0.1`.

## Data layout

Download Boson-nighttime from Hugging Face:

- https://huggingface.co/datasets/xjh19972/boson-nighttime/tree/main/satellite-thermal-dataset-v1

The dataset repository may require a Hugging Face account and approval of the
dataset access conditions before files can be downloaded. The dataset is not
included in this code package.

By default, the code expects datasets outside this source directory:

```text
../datasets/
└── satellite_0_thermalmapping_135_100/
    ├── train_database.h5
    ├── train_queries.h5
    ├── val_database.h5
    ├── val_queries.h5
    ├── test_database.h5
    ├── test_queries.h5
    ├── extended_database.h5
    └── extended_queries.h5
```

Base Dataset training and evaluation use the `train`, `val`, and `test` HDF5
files. Gen Dataset training additionally uses `extended_database.h5` and
`extended_queries.h5`; the latter should correspond to the generated query set
used for the reported TIR-to-RGB setting.

You may use another location by passing `--datasets_folder /path/to/datasets`.
Each HDF5 file is expected to contain:

```text
image_data   # uint8 image patches, shape [N, H, W, 3]
image_size   # image height and width, shape [N, 2]
image_name   # UTF-8 strings containing coordinates, e.g. @y@x
```

For Boson-nighttime, coordinates encoded in `image_name` are used to compute
positive matches and localization error. A retrieval is counted as correct if it
falls within a 50 m radius. Training positives use a 35 m threshold.

Optional checksum reference for the downloaded HDF5 files:

```text
3d2faea188995dd687c77178621b6fc7  extended_database.h5
7ef56acc4a746e7bafe1df91131d7737  test_database.h5
f891d9feca5f4252169a811e8c8e648d  test_queries.h5
f075a7a6db5d5d61be88d252f7d6e05b  train_database.h5
b44ee39173bf356b24690ed6933a6792  train_queries.h5
31923c28dd074ddaacf0c463681f7d2b  val_database.h5
fdcb2e12d9a29b8d20a4cbd88bfe430c  val_queries.h5
```

## Additional datasets used in the paper

The paper also reports experiments on N3C-California, TIMID, and VEDAI. These
datasets are not included in this package; please obtain them from their
original sources and follow the corresponding licenses and access terms.

- **N3C-California**: DSM-RGB retrieval experiments. This dataset is derived
  from the National Agriculture Imagery Program and 3D Elevation Program
  Combined dataset in California introduced by Wang et al. for multimodal
  land-cover semantic segmentation.
  Source: https://github.com/wymqqq/IKDNet-pytorch
- **TIMID**: TIR-MSS retrieval experiments. The Thermal Infrared and
  Multispectral Image Dataset was released through IEEE DataPort.
  Source: https://doi.org/10.21227/Y32J-XG16
- **VEDAI**: zero-shot NIR-RGB transfer evaluation. VEDAI contains paired
  aerial imagery with multiple spectral bands and resolutions; the paper uses
  it to evaluate cross-domain transfer from models trained on Boson-nighttime.
  Source: https://downloads.greyc.fr/vedai/

To use these datasets with this codebase, prepare paired samples in the HDF5
format described above and keep dataset-specific splits and correspondence
files consistent with the evaluation protocol used in the paper.

## Training and evaluation

### Base Dataset training

```bash
bash scripts/train_boson_base_tir2rgb.sh
```

The training parameters are defined at the top of
`scripts/train_boson_base_tir2rgb.sh`.

### Gen Dataset training

```bash
bash scripts/train_boson_gen_tir2rgb.sh
```

This uses `train_extended.py --use_extended_data`, which expects
`extended_database.h5` and `extended_queries.h5` in the dataset directory.

### Evaluation

```bash
bash scripts/eval_boson_tir2rgb.sh
```

The evaluation parameters are defined at the top of
`scripts/eval_boson_tir2rgb.sh`. By default, the script evaluates
`logs/tir2rgb/best_model.pth`. A different checkpoint can be passed as the first
argument:

```bash
bash scripts/eval_boson_tir2rgb.sh logs/tir2rgb_ex/best_model.pth
```

The evaluation prints `R@1`, `R@5`, `R@10`, and `R@20`. The test routine also
computes localization error from coordinates encoded in the HDF5 `image_name`
fields.

You can override the dataset location without editing scripts:

```bash
DATASETS_FOLDER=/path/to/datasets \
DATASET_NAME=satellite_0_thermalmapping_135_100 \
bash scripts/train_boson_base_tir2rgb.sh
```

## Outputs and compute

Training outputs are written to:

```text
logs/DSS-VLAD/<dataset>-<timestamp>-<uuid>/
```

Evaluation outputs are written to:

```text
test/DSS-VLAD/<checkpoint-folder>/<dataset>-<timestamp>/
```

The reported Boson-nighttime runs use approximately:

- Base Dataset training: 14 hours for 100 epochs
- Gen Dataset training: 24 hours for 100 epochs
- peak GPU memory: 4.36 GB

Runtime may vary with CUDA/cuDNN/PyTorch versions, dataloader workers, disk
throughput, and FAISS GPU availability.

## License and third-party notices

This code adapts third-party open-source components. Required copyright notices
and license texts are preserved in `licenses/`:

- `licenses/ARPL-MIT.txt`
- `licenses/REFERENCE-MIT.txt`
- `licenses/PIX2PIX-BSD.txt`
- `licenses/ADDITIONAL_NOTICES.txt`

Included or adapted components include:

- `satellite-thermal-geo-localization`, MIT License
- `deep-visual-geo-localization-benchmark`, MIT License
- `DANN`, MIT License
- `Synchronized-BatchNorm-PyTorch`, MIT License
- `NetVLAD-pytorch`, MIT License
- `pytorch-image-models` / `timm`, Apache License 2.0
- pix2pix / CycleGAN-pix2pix related components, BSD-style licenses

This package does not relicense third-party datasets or model weights. Use all
external assets under their original licenses and terms.
