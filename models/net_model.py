import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.net_common import ConvLayer, ResidualBlock, SEResidualBlock, STNResidualBlock
from models.superglue_gnn import SuperGlue


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
        # the origin version is F.relu
        # conv5 = F.sigmoid(self.conv5(conv4))
        conv5 = F.relu(self.conv5(conv4))

        return conv5


class STNRFNet64(torch.nn.Module):
    def __init__(self):
        super(STNRFNet64, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.stnres1 = STNResidualBlock(128)
        self.stnres2 = STNResidualBlock(128)
        self.stnres3 = STNResidualBlock(128)
        self.stnres4 = STNResidualBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        stnres1 = self.stnres1(conv3)
        stnres2 = self.stnres2(stnres1)
        stnres3 = self.stnres3(stnres2)
        stnres4 = self.stnres4(stnres3)
        conv4 = self.sigmoid(self.conv4(stnres4))
        return conv4


class RFNet64(torch.nn.Module):
    def __init__(self):
        super(RFNet64, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.resid1 = ResidualBlock(128)
        self.resid2 = ResidualBlock(128)
        self.resid3 = ResidualBlock(128)
        self.resid4 = ResidualBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        resid2 = self.resid1(resid1)
        resid3 = self.resid1(resid2)
        resid4 = self.resid1(resid3)
        conv4 = self.sigmoid(self.conv4(resid4))

        # conv4.shape:-> [b, 64, 32, 32]
        return conv4


class SERFNet64(torch.nn.Module):
    def __init__(self):
        super(SERFNet64, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.seres1 = SEResidualBlock(128)
        self.seres2 = SEResidualBlock(128)
        self.seres3 = SEResidualBlock(128)
        self.seres4 = SEResidualBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(conv2))
        seres1 = self.seres1(conv3)
        seres2 = self.seres2(seres1)
        seres3 = self.seres3(seres2)
        seres4 = self.seres4(seres3)
        conv4 = self.sigmoid(self.conv4(seres4))

        # conv4.shape:-> [b, 64, 32, 32]
        return conv4

