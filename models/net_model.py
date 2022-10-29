import torch
import torch.nn as nn
import torch.nn.functional as F
from models.net_common import ConvLayer, ResidualBlock, \
    DeformableConv2d2v


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.zeros_(m.bias)
        torch.nn.init.ones_(m.weight)


class ResidualFeatureNet(torch.nn.Module):
    def __init__(self):
        super(ResidualFeatureNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.resid1 = ResidualBlock(128)
        self.resid2 = ResidualBlock(128)
        self.resid3 = ResidualBlock(128)
        self.resid4 = ResidualBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.conv5 = ConvLayer(64, 1, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        resid2 = self.resid1(resid1)
        resid3 = self.resid1(resid2)
        resid4 = self.resid1(resid3)
        conv4 = F.relu(self.conv4(resid4))
        conv5 = F.relu(self.conv5(conv4))

        return conv5


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(num_features=128)
        self.conv4 = ConvLayer(128, 64, kernel_size=1, stride=1)
        self.bn4 = torch.nn.BatchNorm2d(num_features=64)
        self.conv5 = ConvLayer(64, 1, kernel_size=1, stride=1)
        self.bn5 = torch.nn.BatchNorm2d(num_features=1)

    def forward(self, x):
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        conv3 = F.relu(self.bn3(self.conv3(conv2)))
        conv4 = F.relu(self.bn4(self.conv4(conv3)))
        conv5 = F.relu(self.bn5(self.conv5(conv4)))

        return conv5


class DeConvRFNet(torch.nn.Module):
    def __init__(self):
        super(DeConvRFNet, self).__init__()
        # Initial convolution layers
        self.conv1 = DeformableConv2d2v(3, 32, kernel_size=5, stride=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.conv2 = DeformableConv2d2v(32, 64, kernel_size=3, stride=2, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.conv3 = DeformableConv2d2v(64, 128, kernel_size=3, stride=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(num_features=128)
        self.resid1 = ResidualBlock(128)
        self.resid2 = ResidualBlock(128)
        self.resid3 = ResidualBlock(128)
        self.resid4 = ResidualBlock(128)
        self.conv4 = DeformableConv2d2v(128, 64, kernel_size=3, stride=1)
        self.bn4 = torch.nn.BatchNorm2d(num_features=64)
        self.conv5 = DeformableConv2d2v(64, 1, kernel_size=1, stride=1)
        self.bn5 = torch.nn.BatchNorm2d(num_features=1)

    def forward(self, x):
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        conv3 = F.relu(self.bn3(self.conv3(conv2)))
        resid1 = self.resid1(conv3)
        resid2 = self.resid1(resid1)
        resid3 = self.resid1(resid2)
        resid4 = self.resid1(resid3)
        # conv4 = F.relu((self.bn4(self.conv4(resid4))))
        # conv5 = F.relu(self.bn5(self.conv5(conv4)))
        conv4 = F.relu(self.conv4(resid4))
        conv5 = F.relu(self.conv5(conv4))
        return conv5


class RFNWithSTNet(torch.nn.Module):
    def __init__(self):
        super(RFNWithSTNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(num_features=128)
        self.resid1 = ResidualBlock(128)
        self.resid2 = ResidualBlock(128)
        self.resid3 = ResidualBlock(128)
        self.resid4 = ResidualBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.bn4 = torch.nn.BatchNorm2d(num_features=64)
        self.conv5 = ConvLayer(64, 1, kernel_size=1, stride=1)
        self.bn5 = torch.nn.BatchNorm2d(num_features=1)
        # output shape: [bs, 1, 32, 32]

        # Spatial Transformer Network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def forward(self, x):
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        conv3 = F.relu(self.bn3(self.conv3(conv2)))
        resid1 = self.resid1(conv3)
        resid2 = self.resid1(resid1)
        resid3 = self.resid1(resid2)
        resid4 = self.resid1(resid3)
        conv4 = F.relu(self.bn4(self.conv4(resid4)))
        conv5 = F.relu(self.bn5(self.conv5(conv4)))

        out = self.stn(conv5)

        return out

