import torch
import torch.nn as nn
import torch.nn.functional as F

from models.net_common import ConvLayer, ResidualBlock, SEResidualBlock, STNResidualBlock, \
    DeformResBlock, DilateResBlock, ResBlock, ResBlockRelu, STNResidualBlockRelu


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
        self.conv1 = ConvLayer(1, 32, kernel_size=5, stride=2)
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
        resid2 = self.resid2(resid1)
        resid3 = self.resid3(resid2)
        resid4 = self.resid4(resid3)
        conv4 = F.relu(self.conv4(resid4))
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


class STNResRFNet64(torch.nn.Module):
    def __init__(self):
        super(STNResRFNet64, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.resid1 = ResidualBlock(128)
        self.stnres1 = STNResidualBlock(128)
        self.resid2 = ResidualBlock(128)
        self.stnres2 = STNResidualBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        stnres1 = self.stnres1(resid1)
        resid2 = self.resid2(stnres1)
        stnres2 = self.stnres2(resid2)
        conv4 = self.sigmoid(self.conv4(stnres2))
        return conv4


class STNResRFNet32v216(torch.nn.Module):
    def __init__(self):
        super(STNResRFNet32v216, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 64, kernel_size=3, stride=1)
        self.conv4 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.resid1 = ResidualBlock(128)
        self.stnres1 = STNResidualBlock(128, 16, 16)
        self.stnres2 = STNResidualBlock(128, 16, 16)
        self.stnres3 = STNResidualBlock(128, 16, 16)
        self.conv5 = ConvLayer(128, 32, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv4 = F.relu(self.conv4(conv3))
        resid1 = self.resid1(conv4)
        stnres1 = self.stnres1(resid1)
        stnres2 = self.stnres2(stnres1)
        stnres3 = self.stnres3(stnres2)
        conv5 = self.sigmoid(self.conv5(stnres3))
        return conv5


class STNResRFNet3v216(torch.nn.Module):
    def __init__(self):
        super(STNResRFNet3v216, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 64, kernel_size=3, stride=1)
        self.conv4 = ConvLayer(64, 64, kernel_size=3, stride=2)
        self.resid1 = ResidualBlock(64)
        self.stnres1 = STNResidualBlock(64, 16, 16)
        self.stnres2 = STNResidualBlock(64, 16, 16)
        self.stnres3 = STNResidualBlock(64, 16, 16)
        self.conv5 = ConvLayer(64, 32, kernel_size=3, stride=1)
        self.conv6 = ConvLayer(32, 3, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv4 = F.relu(self.conv4(conv3))
        resid1 = self.resid1(conv4)
        stnres1 = self.stnres1(resid1)
        stnres2 = self.stnres2(stnres1)
        stnres3 = self.stnres3(stnres2)
        conv5 = self.relu(self.conv5(stnres3))
        conv6 = self.sigmoid(self.conv6(conv5))
        return conv6


class STNResRFNet3v232(torch.nn.Module):
    def __init__(self):
        super(STNResRFNet3v232, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 64, kernel_size=3, stride=1)
        self.resid1 = ResidualBlock(64)
        self.stnres1 = STNResidualBlock(64, 32, 32)
        self.stnres2 = STNResidualBlock(64, 32, 32)
        self.stnres3 = STNResidualBlock(64, 32, 32)
        self.conv4 = ConvLayer(64, 32, kernel_size=3, stride=1)
        self.conv5 = ConvLayer(32, 3, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        stnres1 = self.stnres1(resid1)
        stnres2 = self.stnres2(stnres1)
        stnres3 = self.stnres3(stnres2)
        conv4 = self.relu(self.conv4(stnres3))
        conv5 = self.sigmoid(self.conv5(conv4))
        return conv5


class STNResRFNet32v316(torch.nn.Module):
    def __init__(self):
        super(STNResRFNet32v316, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 32, kernel_size=3, stride=1)
        self.resid1 = ResidualBlock(32)
        self.conv3 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv4 = ConvLayer(64, 64, kernel_size=3, stride=1)
        self.stnres1 = STNResidualBlock(64, 32, 32)
        self.conv5 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.stnres2 = STNResidualBlock(128, 16, 16)
        self.stnres3 = STNResidualBlock(128, 16, 16)
        self.conv6 = ConvLayer(128, 32, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        resid1 = self.resid1(conv2)
        conv3 = self.relu(self.conv3(resid1))
        conv4 = self.relu(self.conv4(conv3))
        stnres1 = self.stnres1(conv4)
        conv5 = self.relu(self.conv5(stnres1))
        stnres2 = self.stnres2(conv5)
        stnres3 = self.stnres3(stnres2)
        conv6 = self.sigmoid(self.conv6(stnres3))
        return conv6


class STNResRFNet3v316(torch.nn.Module):
    def __init__(self):
        super(STNResRFNet3v316, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 32, kernel_size=3, stride=1)
        self.resid1 = ResidualBlock(32)
        self.conv3 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv4 = ConvLayer(64, 64, kernel_size=3, stride=1)
        self.stnres1 = STNResidualBlock(64, 32, 32)
        self.conv5 = ConvLayer(64, 64, kernel_size=3, stride=2)
        self.stnres2 = STNResidualBlock(64, 16, 16)
        self.stnres3 = STNResidualBlock(64, 16, 16)
        self.conv6 = ConvLayer(64, 32, kernel_size=3, stride=1)
        self.conv7 = ConvLayer(32, 3, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        resid1 = self.resid1(conv2)
        conv3 = self.relu(self.conv3(resid1))
        conv4 = self.relu(self.conv4(conv3))
        stnres1 = self.stnres1(conv4)
        conv5 = self.relu(self.conv5(stnres1))
        stnres2 = self.stnres2(conv5)
        stnres3 = self.stnres3(stnres2)
        conv6 = self.relu(self.conv6(stnres3))
        conv7 = self.sigmoid(self.conv7(conv6))
        return conv7


class STNResRFNet3v332(torch.nn.Module):
    def __init__(self):
        super(STNResRFNet3v332, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 32, kernel_size=3, stride=1)
        self.resid1 = ResidualBlock(32)
        self.conv3 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv4 = ConvLayer(64, 64, kernel_size=3, stride=1)
        self.stnres1 = STNResidualBlock(64, 32, 32)
        self.conv5 = ConvLayer(64, 64, kernel_size=3, stride=1)
        self.stnres2 = STNResidualBlock(64, 32, 32)
        self.stnres3 = STNResidualBlock(64, 32, 32)
        self.conv6 = ConvLayer(64, 32, kernel_size=3, stride=1)
        self.conv7 = ConvLayer(32, 3, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        resid1 = self.resid1(conv2)
        conv3 = self.relu(self.conv3(resid1))
        conv4 = self.relu(self.conv4(conv3))
        stnres1 = self.stnres1(conv4)
        conv5 = self.relu(self.conv5(stnres1))
        stnres2 = self.stnres2(conv5)
        stnres3 = self.stnres3(stnres2)
        conv6 = self.relu(self.conv6(stnres3))
        conv7 = self.sigmoid(self.conv7(conv6))
        return conv7


class STNResRFNet64v2(torch.nn.Module):
    def __init__(self):
        super(STNResRFNet64v2, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.resid1 = ResidualBlock(128)
        self.stnres1 = STNResidualBlock(128)
        self.stnres2 = STNResidualBlock(128)
        self.stnres3 = STNResidualBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        stnres1 = self.stnres1(resid1)
        stnres2 = self.stnres2(stnres1)
        stnres3 = self.stnres3(stnres2)
        conv4 = self.sigmoid(self.conv4(stnres3))
        return conv4


class STNResRFNet64v2Relu(torch.nn.Module):
    def __init__(self):
        super(STNResRFNet64v2Relu, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.resid1 = ResBlockRelu(128)
        self.stnres1 = STNResidualBlockRelu(128)
        self.stnres2 = STNResidualBlockRelu(128)
        self.stnres3 = STNResidualBlockRelu(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        stnres1 = self.stnres1(resid1)
        stnres2 = self.stnres2(stnres1)
        stnres3 = self.stnres3(stnres2)
        conv4 = self.sigmoid(self.conv4(stnres3))
        return conv4


class STNResRFNet64v3(torch.nn.Module):
    def __init__(self):
        super(STNResRFNet64v3, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.resid1 = ResidualBlock(128)
        self.stnres1 = STNResidualBlock(128)
        self.stnres2 = STNResidualBlock(128)
        self.resid2 = ResidualBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        stnres1 = self.stnres1(resid1)
        stnres2 = self.stnres2(stnres1)
        resid2 = self.resid2(stnres2)
        conv4 = self.sigmoid(self.conv4(resid2))
        return conv4


class STNResRFNet64v4(torch.nn.Module):
    def __init__(self):
        super(STNResRFNet64v4, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.resid1 = ResidualBlock(128)
        self.stnres1 = STNResidualBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        stnres1 = self.stnres1(resid1)
        conv4 = self.sigmoid(self.conv4(stnres1))
        return conv4


class STNResRFNet64v5(torch.nn.Module):
    def __init__(self):
        super(STNResRFNet64v5, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.stnres1 = STNResidualBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        stnres1 = self.stnres1(conv3)
        conv4 = self.sigmoid(self.conv4(stnres1))
        return conv4


class RFNet64(torch.nn.Module):
    def __init__(self):
        super(RFNet64, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.resid1 = ResBlock(128)
        self.resid2 = ResBlock(128)
        self.resid3 = ResBlock(128)
        self.resid4 = ResBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        resid2 = self.resid2(resid1)
        resid3 = self.resid3(resid2)
        resid4 = self.resid4(resid3)
        conv4 = self.sigmoid(self.conv4(resid4))

        # conv4.shape:-> [b, 64, 32, 32]
        return conv4


class RFNet64Relu(torch.nn.Module):
    def __init__(self):
        super(RFNet64Relu, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.resid1 = ResBlockRelu(128)
        self.resid2 = ResBlockRelu(128)
        self.resid3 = ResBlockRelu(128)
        self.resid4 = ResBlockRelu(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        resid2 = self.resid2(resid1)
        resid3 = self.resid3(resid2)
        resid4 = self.resid4(resid3)
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


class DeformRFNet64(torch.nn.Module):
    def __init__(self):
        super(DeformRFNet64, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.resid1 = ResidualBlock(128)
        self.resid2 = DeformResBlock(128)
        self.resid3 = DeformResBlock(128)
        self.resid4 = DeformResBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        resid2 = self.resid2(resid1)
        resid3 = self.resid3(resid2)
        resid4 = self.resid4(resid3)
        conv4 = self.sigmoid(self.conv4(resid4))

        # conv4.shape:-> [b, 64, 32, 32]
        return conv4


class DilateRFNet64(torch.nn.Module):
    def __init__(self):
        super(DilateRFNet64, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.resid1 = ResidualBlock(128)
        self.resid2 = DilateResBlock(128)
        self.resid3 = DilateResBlock(128)
        self.resid4 = DilateResBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        resid2 = self.resid2(resid1)
        resid3 = self.resid3(resid2)
        resid4 = self.resid4(resid3)
        conv4 = self.sigmoid(self.conv4(resid4))

        # conv4.shape:-> [b, 64, 32, 32]
        return conv4
