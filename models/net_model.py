import torch
import torch.nn as nn
import torch.nn.functional as F
from models.net_common import ConvLayer, ResidualBlock, \
    DeformableConv2d2v
from EfficientNetV2 import


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


class AssistantModel(torch.nn.Module):
    def __init__(self):
        super(AssistantModel, self).__init__()
        # ======================================== reduce dimension
        self.reduce1 = nn.Conv2d(in_channels=1280, out_channels=640, kernel_size=1)
        self.reduce2 = nn.Conv2d(in_channels=640, out_channels=320, kernel_size=1)
        self.reduce3 = nn.Conv2d(in_channels=320, out_channels=128, kernel_size=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels=640, out_channels=320, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=320)
        self.conv2 = nn.Conv2d(in_channels=320, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.resid1 = ResidualBlock(64)
        self.resid2 = ResidualBlock(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3)

    def forward(self, x8, x16, x32):
        x32 = F.sigmoid(self.reduce1(x32))
        x16 = F.sigmoid(self.reduce2(x16))
        x8 = F.sigmoid(self.reduce3(x8))

        x32 = self.upsample(x32)
        x32 = F.relu(self.bn1(self.conv1(x32)))
        x16 = x16+x32
        x16 = self.upsample(x16)
        x16 = F.relu(self.bn2(self.conv2(x16)))
        x8 = x8 + x16
        out = F.relu(self.bn3(self.conv3(x8)))
        out = self.resid1(out)
        out = self.resid2(out)
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.relu(self.conv5(out))

        return out


class FusionModel(torch.nn.Module):
    """
    Fuse segmented finger knuckle features and yolov5's output feature
    """
    def __init__(self):
        super(FusionModel, self).__init__()
        # fuse yolov5 features
        self.reduce1 = nn.Conv2d(in_channels=1280, out_channels=640, kernel_size=1)
        self.reduce2 = nn.Conv2d(in_channels=640, out_channels=320, kernel_size=1)
        self.reduce3 = nn.Conv2d(in_channels=320, out_channels=160, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels=640, out_channels=320, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=320)
        self.conv2 = nn.Conv2d(in_channels=320, out_channels=160, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=160)
        self.conv3 = nn.Conv2d(in_channels=160, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)

        # extract finger knuckle features


        # fuse feature maps by adding



    def forward(self, x, s8, s16, s32):
        # fuse yolov5 features
        s32 = F.sigmoid(self.reduce1(s32))
        s16 = F.sigmoid(self.reduce2(s16))
        s8 = F.sigmoid(self.reduce3(s8))
        s32 = self.upsample(s32)
        s32 = F.relu(self.bn1(self.conv1(s32)))
        s16 = s32 + s16
        s16 = self.upsample(s16)
        s16 = F.relu(self.bn2(self.conv2(s16)))
        s8 = s8 + s16
        s8 = F.relu(self.bn3(self.conv3(s8)))






