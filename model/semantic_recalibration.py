import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm
from timm.models.layers import DropPath


class DynamicSemanticRecalibration(nn.Module):
    def __init__(self, C, reduction=4, drop_path=0.2, temp=4.0):
        super().__init__()
        self.C = C
        hidden = max(C // reduction, 1)
        self.temp = nn.Parameter(torch.ones(1) * np.log(temp))
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            spectral_norm(nn.Conv1d(C * 2, hidden, 1)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv1d(hidden, C, 1)),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, dc):
        dc_trans = dc.transpose(1, 2)
        avg_out = self.avg_pool(dc_trans)
        max_out = self.max_pool(dc_trans)
        combined = torch.cat([avg_out, max_out], dim=1)
        w = self.fc(combined) / torch.exp(self.temp)
        w = torch.sigmoid(w)
        if self.training:
            w = self.drop_path(w)
        return (w * dc_trans).transpose(1, 2)


class SEGate(nn.Module):
    def __init__(self, C, reduction=4):
        super().__init__()
        hidden = max(C // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(C, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, C, 1),
            nn.Sigmoid(),
        )

    def forward(self, dc):
        dc_trans = dc.transpose(1, 2)
        w = self.fc(self.avg_pool(dc_trans))
        return (w * dc_trans).transpose(1, 2)


class CBAMGate(nn.Module):
    def __init__(self, C, reduction=4, kernel_size=7):
        super().__init__()
        hidden = max(C // reduction, 1)
        padding = kernel_size // 2
        self.mlp = nn.Sequential(
            nn.Conv1d(C, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, C, 1),
        )
        self.spatial = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, dc):
        dc_trans = dc.transpose(1, 2)
        avg_out = self.mlp(torch.mean(dc_trans, dim=2, keepdim=True))
        max_out = self.mlp(torch.amax(dc_trans, dim=2, keepdim=True))
        channel_w = torch.sigmoid(avg_out + max_out)
        dc_trans = channel_w * dc_trans

        avg_out = torch.mean(dc_trans, dim=1, keepdim=True)
        max_out = torch.amax(dc_trans, dim=1, keepdim=True)
        spatial_w = self.spatial(torch.cat([avg_out, max_out], dim=1))
        return (spatial_w * dc_trans).transpose(1, 2)


def build_vlad_gate(gate_type, C, drop_path=0.2):
    if gate_type in (None, "none"):
        return nn.Identity()
    if gate_type == "learnable_temp_srcg":
        return DynamicSemanticRecalibration(C=C, drop_path=drop_path)
    if gate_type == "se":
        return SEGate(C=C)
    if gate_type == "cbam":
        return CBAMGate(C=C)
    raise ValueError(f"Unsupported gate_type: {gate_type}")
