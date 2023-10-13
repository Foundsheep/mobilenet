from torch import nn

from typing import Union


class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        # --- reference ---
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # -----------------
        self.depthwise_layer = nn.Conv2d(in_channels=n_in, out_channels=n_in, kernel_size=3, padding=1, groups=n_in)
        self.bn_1 = nn.BatchNorm2d(n_in)
        self.relu_1 = nn.ReLU()
        self.pointwise_layer = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=1)
        self.bn_2 = nn.BatchNorm2d(n_out)
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_layer(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.pointwise_layer(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        return x


class MobileNet(nn.Module):
    def __int__(self, width_multiplier: Union[int, float], resolution_multiplier: Union[int, float], is_mobile: bool):
        self.ga = nn.AvgPool2d()
        self.fc = nn.Linear()
        pass

    def forward(self):
        pass