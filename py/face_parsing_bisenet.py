"""Minimal facexlib-compatible BiSeNet face parsing model.

The bundled ``parsing_bisenet.pth`` weights use the original facexlib module
names.  Keeping the small network definition here avoids making facexlib a
runtime dependency of CineStyle.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _BasicBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, stride: int = 1):
        super().__init__()
        self.conv1 = _conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = _conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        shortcut = x if self.downsample is None else self.downsample(x)
        return self.relu(shortcut + residual)


def _make_layer(in_chan: int, out_chan: int, blocks: int, stride: int = 1) -> nn.Sequential:
    layers = [_BasicBlock(in_chan, out_chan, stride)]
    layers.extend(_BasicBlock(out_chan, out_chan) for _ in range(blocks - 1))
    return nn.Sequential(*layers)


class _ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _make_layer(64, 64, 2)
        self.layer2 = _make_layer(64, 128, 2, stride=2)
        self.layer3 = _make_layer(128, 256, 2, stride=2)
        self.layer4 = _make_layer(256, 512, 2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32


class _ConvBNReLU(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, ks: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class _AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan: int, out_chan: int):
        super().__init__()
        self.conv = _ConvBNReLU(in_chan, out_chan)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.sigmoid_atten(self.bn_atten(self.conv_atten(atten)))
        return feat * atten


class _ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = _ResNet18()
        self.arm16 = _AttentionRefinementModule(256, 128)
        self.arm32 = _AttentionRefinementModule(512, 128)
        self.conv_head32 = _ConvBNReLU(128, 128)
        self.conv_head16 = _ConvBNReLU(128, 128)
        self.conv_avg = _ConvBNReLU(512, 128, ks=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat8, feat16, feat32 = self.resnet(x)
        h16, w16 = feat16.size()[2:]
        h32, w32 = feat32.size()[2:]
        avg = self.conv_avg(F.avg_pool2d(feat32, feat32.size()[2:]))
        avg_up = F.interpolate(avg, (h32, w32), mode="nearest")
        feat32_up = self.conv_head32(F.interpolate(self.arm32(feat32) + avg_up, (h16, w16), mode="nearest"))
        feat16_up = self.conv_head16(F.interpolate(self.arm16(feat16) + feat32_up, feat8.size()[2:], mode="nearest"))
        return feat8, feat16_up, feat32_up


class _FeatureFusionModule(nn.Module):
    def __init__(self, in_chan: int, out_chan: int):
        super().__init__()
        self.convblk = _ConvBNReLU(in_chan, out_chan, ks=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp: torch.Tensor, fcp: torch.Tensor) -> torch.Tensor:
        feat = self.convblk(torch.cat([fsp, fcp], dim=1))
        atten = self.sigmoid(self.conv2(self.relu(self.conv1(F.avg_pool2d(feat, feat.size()[2:])))))
        return feat * atten + feat


class _BiSeNetOutput(nn.Module):
    def __init__(self, in_chan: int, mid_chan: int, num_class: int):
        super().__init__()
        self.conv = _ConvBNReLU(in_chan, mid_chan)
        self.conv_out = nn.Conv2d(mid_chan, num_class, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        return self.conv_out(feat), feat


class BiSeNet(nn.Module):
    def __init__(self, num_class: int = 19):
        super().__init__()
        self.cp = _ContextPath()
        self.ffm = _FeatureFusionModule(256, 256)
        self.conv_out = _BiSeNetOutput(256, 256, num_class)
        self.conv_out16 = _BiSeNetOutput(128, 64, num_class)
        self.conv_out32 = _BiSeNetOutput(128, 64, num_class)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)
        out = self.conv_out(feat_fuse)[0]
        out16 = self.conv_out16(feat_cp8)[0]
        out32 = self.conv_out32(feat_cp16)[0]
        size = (height, width)
        return (
            F.interpolate(out, size, mode="bilinear", align_corners=True),
            F.interpolate(out16, size, mode="bilinear", align_corners=True),
            F.interpolate(out32, size, mode="bilinear", align_corners=True),
        )
