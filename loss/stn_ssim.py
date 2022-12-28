import torch
from models.pytorch_mssim import SSIM
from models.net_common import STN
import torch.nn.functional as F


class STN(torch.nn.Module):
    def __init__(self, input_channels, input_h, input_w):
        super(STN, self).__init__()
        self.input_channels = input_channels
        self.input_h = input_h
        self.input_w = input_w

        self.localization = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, input_channels, kernel_size=5),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(input_channels, input_channels, kernel_size=3),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.ReLU(True)
        )
        self.fc_loc = torch.nn.Sequential(
            torch.nn.Linear(int(input_channels * (input_h / 4 - 2) * (input_w / 4 - 2)),
                            int(input_channels * (input_h / 4 - 2) * (input_w / 4 - 2))),
            # torch.nn.ReLU(True),
            torch.nn.Tanh(),
            torch.nn.Linear(int(input_channels * (input_h / 4 - 2) * (input_w / 4 - 2)),
                            int(input_channels * (input_h / 4 - 2))),
            # torch.nn.ReLU(True),
            torch.nn.Tanh(),
            torch.nn.Linear(int(input_channels * (input_h / 4 - 2)), input_channels),
            # torch.nn.ReLU(True),
            torch.nn.Tanh(),
            torch.nn.Linear(input_channels, 3 * 2),
            # new add
            torch.nn.Tanh()
        )

        self.fc_loc[6].weight.data.zero_()
        self.fc_loc[6].bias.data.copy_(torch.tensor([3, 0, 0, 0, 3, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, int(self.input_channels * (self.input_h / 4 - 2) * (self.input_w / 4 - 2)))
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        return theta


class STSSIM(torch.nn.Module):
    def __init__(self, data_range=1.0, size_average=False, channel=64):
        super(STSSIM, self).__init__()
        self.data_range = data_range
        self.size_average = size_average
        self.channel = channel
        self.ssim = SSIM(data_range=self.data_range, size_average=self.size_average, channel=self.channel)
        self.stn = STN(input_channels=int(self.channel * 2), input_h=32, input_w=32)

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        theta = self.stn(out)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        ssim = self.ssim(x, y)
        return ssim

