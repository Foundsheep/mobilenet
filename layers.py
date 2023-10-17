from torch import nn
from torchvision import transforms

from typing import Union


class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        # --- reference ---
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # -----------------
        self.depthwise_layer = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding="same" if stride > 1 else "valid", groups=in_channels, stride=stride)
        self.bn_1 = nn.BatchNorm2d(in_channels)
        self.relu_1 = nn.ReLU()
        self.pointwise_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu_2 = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        x = self.depthwise_layer(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.pointwise_layer(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        return x


class MobileNet(nn.Module):
    def __int__(self, width_multiplier, resolution_multiplier=1, is_mobile=True):
        self.resolution_multiplier = resolution_multiplier

        base_conv = nn.Conv2d if is_mobile else DepthwiseSeparableConvolution
        self.conv_01 = nn.Conv2d(in_channels=3, out_channels=int(32 * width_multiplier), kernel_size=3, stride=2)
        self.dp_layer_02 = base_conv(in_channels=self.conv_01.out_channels, out_channels=self.conv_01.out_channels * 2, stride=1)
        self.dp_layer_03 = base_conv(in_channels=self.dp_layer_02.out_channels, out_channels=self.dp_layer_02.out_channels * 2, stride=2)
        self.dp_layer_04 = base_conv(in_channels=self.dp_layer_03.out_channels, out_channels=self.dp_layer_03.out_channels * 2, stride=1)
        self.dp_layer_05 = base_conv(in_channels=self.dp_layer_04.out_channels, out_channels=self.dp_layer_04.out_channels * 2, stride=2)
        self.dp_layer_06 = base_conv(in_channels=self.dp_layer_05.out_channels, out_channels=self.dp_layer_05.out_channels * 2, stride=1)
        self.dp_layer_07 = base_conv(in_channels=self.dp_layer_06.out_channels, out_channels=self.dp_layer_06.out_channels * 2, stride=2)
        self.dp_layer_list = nn.ModuleList(
            [base_conv(in_channels=self.dp_layer_07.out_channels, out_channels=self.dp_layer_07.out_channels, stride=1) for _ in range(5)]
        )
        self.dp_layer_13 = base_conv(in_channels=self.dp_layer_07.out_channels, out_channels=self.dp_layer_07.out_channels * 2, stride=2)
        self.dp_layer_14 = base_conv(in_channels=self.dp_layer_13.out_channels, out_channels=self.dp_layer_13.out_channels * 2, stride=2)
        self.ga = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(in_features=1024, out_features=1000)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if self.resolution_multiplier != 1:
            C, H, W = x.size()
            x = transforms.Resize(int(H*self.resolution_multiplier))(x)
        x = self.conv_01(x)
        x = self.dp_layer_02(x)
        x = self.dp_layer_03(x)
        x = self.dp_layer_04(x)
        x = self.dp_layer_05(x)
        x = self.dp_layer_06(x)
        x = self.dp_layer_07(x)
        for layer in self.dp_layer_list:
            x = layer(x)
        x = self.dp_layer_13(x)
        x = self.dp_layer_14(x)
        x = self.ga(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
