import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ExplicitSpatialPrior(nn.Module):
    def __init__(self, in_channels, coord_type='none', mask_mode='multiplication', coordconv=False):
        super().__init__()
        
        self.coord_type = coord_type.lower()
        self.mask_mode = mask_mode.lower()
        self.coordconv = coordconv
        
        if self.coord_type == 'xy' or self.coord_type == 'gauss':
            extra_channels = 2
        elif self.coord_type in ['x', 'y']:
            extra_channels = 1
        else:
            extra_channels = 0
        
        if self.coordconv:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels + extra_channels, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels)
            )
        else:
            self.embedding = nn.Sequential(
                nn.Conv2d(in_channels + extra_channels, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        b, c, h, w = x.size()
        features_to_cat = [x]

        if self.coord_type in ['x', 'xy', 'gauss']:
            x_grid = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
            if self.coord_type == 'gauss':
                sigma = 0.5
                x_grid = torch.exp(-(x_grid ** 2) / (2 * (sigma ** 2)))
            features_to_cat.append(x_grid)
        
        if self.coord_type in ['y', 'xy', 'gauss']:
            y_grid = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
            if self.coord_type == 'gauss':
                sigma = 0.5
                y_grid = torch.exp(-(y_grid ** 2) / (2 * (sigma ** 2)))
            features_to_cat.append(y_grid)

        if len(features_to_cat) > 1:
            x_coords = torch.cat(features_to_cat, dim=1)
        else:
            x_coords = features_to_cat[0]
        
        if self.coordconv:
            coord_x = self.proj(x_coords)
            return coord_x
        else:
            spatial_prior = self.embedding(x_coords)

            if self.mask_mode == 'addition':
                return x + spatial_prior
            elif self.mask_mode == 'multiplication':
                return x * spatial_prior
            elif self.mask_mode == 'concatenation':
                return torch.cat([x, spatial_prior], dim=1)
            else:
                raise ValueError(f"Unsupported mask_mode: {self.mask_mode}. Choose from 'addition', 'multiplication', or 'concatenation'.")


class AttentionMap(nn.Module):
    """Analyze the effect of spatial prior by attention mechanism
    """
    def __init__(self, in_channels, feat_size=32):
        super().__init__()

        self.pos_embed = Parameter(
            0.02 * torch.randn(1, in_channels, feat_size, feat_size)
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.bg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, in_channels),
        )

    def forward(self, x):
        pos_embed = self.pos_embed
        if pos_embed.shape[-2:] != x.shape[-2:]:
            pos_embed = F.interpolate(
                pos_embed,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        spatial_input = x + pos_embed
        spatial_weights = self.spatial_conv(spatial_input)

        bg_weights = 1.0 - spatial_weights
        bg_features = self.bg_conv(x)

        return x * spatial_weights + bg_features * bg_weights
     


class ExplicitSemanticEnhancement(nn.Module):
    """DoubleBranch-semanticspatialprior
    """
    def __init__(self, in_channels):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels + 2, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

        self.explicit_semantic = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.GroupNorm(8, in_channels),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y_grid = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        x_grid = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        coord_map = torch.cat([x_grid, y_grid], dim=1)
        x_coords = torch.cat([coord_map, x], dim=1)
        spatial_prior = self.embedding(x_coords)

        explicit_semantic = self.explicit_semantic(x)

        return x * spatial_prior + explicit_semantic



