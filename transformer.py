import torch
import torch.nn as nn
import torch.nn.functional as F


def Conv(in_s, out_s, k=3, stride=1):
    return nn.Sequential(
        nn.ReflectionPad2d(padding=k // 2),
        nn.Conv2d(in_s, out_s, k, stride)
    )


def ConvNormReLU(in_s, out_s, k=3, stride=1):
    return nn.Sequential(
        nn.ReflectionPad2d(padding=k // 2),
        nn.Conv2d(in_s, out_s, k, stride),
        nn.InstanceNorm2d(out_s, affine=True),
        nn.ReLU(inplace=True)
    )


class ResBlock(nn.Module):
    def __init__(self, block):
        super(ResBlock, self).__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


def Res(in_s):
    return ResBlock(nn.Sequential(
        ConvNormReLU(in_s, in_s),
        Conv(in_s, in_s),
        nn.InstanceNorm2d(in_s, affine=True)
    ))


def Upsample(in_s, out_s, up=2, k=3, s=1):
    return nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=up),
        Conv(in_s, out_s, k, s),
        nn.InstanceNorm2d(out_s, affine=True),
        nn.ReLU(inplace=True)
    )


def Transformer():
    return nn.Sequential(
        ConvNormReLU(3, 32, 9),
        ConvNormReLU(32, 64, 3, 2),
        ConvNormReLU(64, 128, 3, 2),
        Res(128),
        Res(128),
        Res(128),
        Res(128),
        Res(128),
        Upsample(128, 64),
        Upsample(64, 64),
        Upsample(64, 32),
        Conv(32, 3, 9),
        nn.Sigmoid()
    )
