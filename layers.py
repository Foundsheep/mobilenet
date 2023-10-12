from torch import nn


class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.depthwise_layer = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=3, padding=1, groups=n_in)
        self.pointwise_layer = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=1)

    def forward(self, x):
        x = self.depthwise_layer(x)
        x = self.pointwise_layer(x)
        return x

