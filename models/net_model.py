import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.net_common import ConvLayer, ResidualBlock, \
    DeformableConv2d2v, ResWithSTNBlock
from models.EfficientNetV2 import fk_efficientnetv2_s_nohead
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
        self.conv6 = ConvLayer

    def forward(self, x, mask):
        mask = mask
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        resid2 = self.resid1(resid1)
        resid3 = self.resid1(resid2)
        resid4 = self.resid1(resid3)
        conv4 = F.relu(self.conv4(resid4))
        # the origin version is F.relu
        conv5 = F.sigmoid(self.conv5(conv4))
        # conv5 = F.relu(self.conv5(conv4))

        return conv5, mask


class ResidualSTNet(torch.nn.Module):
    def __init__(self):
        super(ResidualSTNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        # self.resid1 = ResidualBlock(128)
        # self.resid2 = ResidualBlock(128)
        # self.resid3 = ResidualBlock(128)
        # self.resid4 = ResidualBlock(128)
        self.resid1 = ResWithSTNBlock(128)
        self.resid2 = ResWithSTNBlock(128)
        self.resid3 = ResWithSTNBlock(128)
        self.resid4 = ResWithSTNBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.conv5 = ConvLayer(64, 1, kernel_size=1, stride=1)

    def forward(self, x, mask):
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

    def forward(self, x, mask):
        mask = mask
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        resid2 = self.resid1(resid1)
        resid3 = self.resid1(resid2)
        resid4 = self.resid1(resid3)
        conv4 = F.sigmoid(self.conv4(resid4))

        # conv4.shape:-> [b, 64, 32, 32]
        return conv4, mask


class RFNet64GNN(torch.nn.Module):
    def __init__(self):
        super(RFNet64GNN, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.resid1 = ResidualBlock(128)
        self.resid2 = ResidualBlock(128)
        self.resid3 = ResidualBlock(128)
        self.resid4 = ResidualBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.gnn = SuperGlue

    def forward(self, x, mask):
        mask = mask
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        resid2 = self.resid1(resid1)
        resid3 = self.resid1(resid2)
        resid4 = self.resid1(resid3)
        conv4 = F.relu(self.conv4(resid4))

        # conv4.shape:-> [b, 64, 32, 32]
        return conv4, mask

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
            nn.Conv2d(1, 16, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 6 * 6, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # for m in self.modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
        #         if m.bias is not None:
        #             torch.nn.init.zeros_(m.bias)
        #     elif isinstance(m, torch.nn.BatchNorm2d):
        #         nn.init.ones_(m.weight)
        #         nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         nn.init.zeros_(m.bias)

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x, mask):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 6 * 6)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        mask = F.grid_sample(mask, grid, align_corners=True)

        return x, mask

    def forward(self, x, mask):
        # conv1 = F.relu(self.bn1(self.conv1(x)))
        # conv2 = F.relu(self.bn2(self.conv2(conv1)))
        # conv3 = F.relu(self.bn3(self.conv3(conv2)))
        # resid1 = self.resid1(conv3)
        # resid2 = self.resid1(resid1)
        # resid3 = self.resid1(resid2)
        # resid4 = self.resid1(resid3)
        # conv4 = F.relu(self.bn4(self.conv4(resid4)))
        # conv5 = F.relu(self.bn5(self.conv5(conv4)))

        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        resid2 = self.resid1(resid1)
        resid3 = self.resid1(resid2)
        resid4 = self.resid1(resid3)
        conv4 = F.relu(self.conv4(resid4))
        conv5 = F.relu(self.conv5(conv4))

        out, mask = self.stn(conv5, mask)

        return out, mask


class STNWithRFNet(torch.nn.Module):
    def __init__(self):
        super(STNWithRFNet, self).__init__()
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
            nn.Conv2d(3, 16, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 30 * 30, 32 * 30),
            nn.ReLU(True),
            nn.Linear(32 * 30, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # for m in self.modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
        #         if m.bias is not None:
        #             torch.nn.init.zeros_(m.bias)
        #     elif isinstance(m, torch.nn.BatchNorm2d):
        #         nn.init.ones_(m.weight)
        #         nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         nn.init.zeros_(m.bias)

        self.fc_loc[4].weight.data.zero_()
        self.fc_loc[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x, mask):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 30 * 30)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        # the translation of mask should be a quarter of input image
        mask_theta = theta[:, :, -1] / 4
        theta[:, :, -1] = mask_theta
        grid = F.affine_grid(theta, mask.size(), align_corners=True)
        mask = F.grid_sample(mask, grid, align_corners=True)

        return x, mask

    def forward(self, x, mask):
        x, mask = self.stn(x, mask)
        # conv1 = F.relu(self.bn1(self.conv1(x)))
        # conv2 = F.relu(self.bn2(self.conv2(conv1)))
        # conv3 = F.relu(self.bn3(self.conv3(conv2)))
        # resid1 = self.resid1(conv3)
        # resid2 = self.resid1(resid1)
        # resid3 = self.resid1(resid2)
        # resid4 = self.resid1(resid3)
        # conv4 = F.relu(self.bn4(self.conv4(resid4)))
        # conv5 = F.relu(self.bn5(self.conv5(conv4)))

        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        resid2 = self.resid1(resid1)
        resid3 = self.resid1(resid2)
        resid4 = self.resid1(resid3)
        conv4 = F.relu(self.conv4(resid4))
        conv5 = F.relu(self.conv5(conv4))

        return conv5, mask


class AssistantModel(torch.nn.Module):
    def __init__(self):
        super(AssistantModel, self).__init__()
        # ======================================== reduce dimension
        self.reduce1 = nn.Conv2d(in_channels=1280, out_channels=640, kernel_size=1)
        self.reduce1bn = nn.BatchNorm2d(num_features=640)
        self.reduce2 = nn.Conv2d(in_channels=640, out_channels=320, kernel_size=1)
        self.reduce2bn = nn.BatchNorm2d(num_features=320)
        self.reduce3 = nn.Conv2d(in_channels=320, out_channels=128, kernel_size=1)
        self.reduce3bn = nn.BatchNorm2d(128)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels=640, out_channels=320, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_features=320)
        self.conv2 = nn.Conv2d(in_channels=320, out_channels=128, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.resid1 = ResidualBlock(128)
        self.resid2 = ResidualBlock(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3)

    def forward(self, x8, x16, x32):
        # x8.shape():-> [b, 320, 22, 26]
        # x16.shape():-> [b, 640, 12, 14]
        # x32.shape():-> [b, 1280, 7, 8]
        x32 = F.sigmoid(self.reduce1bn(self.reduce1(x32)))
        x16 = F.sigmoid(self.reduce2bn(self.reduce2(x16)))
        x8 = F.sigmoid(self.reduce3bn(self.reduce3(x8)))

        x32 = self.upsample(x32)
        x32 = F.relu(self.bn1(self.conv1(x32)))
        x16 = x16 + x32
        x16 = self.upsample(x16)
        x16 = F.relu(self.bn2(self.conv2(x16)))
        x8 = x8 + x16
        out = F.relu(self.bn3(self.conv3(x8)))
        out = self.resid1(out)
        out = self.resid2(out)
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.relu(self.conv5(out))
        # out.shape():-> [b, c, 16, 20]

        return out


class FusionModel(torch.nn.Module):
    """
    Fuse segmented finger knuckle features and yolov5's output feature
    """

    def __init__(self):
        super(FusionModel, self).__init__()
        # fuse yolov5 features
        self.reduce1 = nn.Conv2d(in_channels=1280, out_channels=640, kernel_size=1)
        self.reduce1bn = nn.BatchNorm2d(num_features=640)
        self.reduce2 = nn.Conv2d(in_channels=640, out_channels=320, kernel_size=1)
        self.reduce2bn = nn.BatchNorm2d(num_features=320)
        self.reduce3 = nn.Conv2d(in_channels=320, out_channels=128, kernel_size=1)
        self.reduce3bn = nn.BatchNorm2d(num_features=128)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels=640, out_channels=320, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_features=320)
        self.conv2 = nn.Conv2d(in_channels=320, out_channels=128, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)

        # extract finger knuckle features
        self.fknet = RFNet64()

        self.middle = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1)
        self.middlebn = nn.BatchNorm2d(num_features=128)
        # fuse feature maps by add
        self.resid1 = ResidualBlock(128)
        self.resid2 = ResidualBlock(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(num_features=32)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3)

    def forward(self, x, s8, s16, s32):
        # fuse yolov5 features output [b, 64, 22, 26]
        s32 = F.sigmoid(self.reduce1bn(self.reduce1(s32)))
        s16 = F.sigmoid(self.reduce2bn(self.reduce2(s16)))
        s8 = F.sigmoid(self.reduce3bn(self.reduce3(s8)))
        s32 = self.upsample(s32)
        s32 = F.relu(self.bn1(self.conv1(s32)))
        s16 = s32 + s16
        s16 = self.upsample(s16)
        s16 = F.relu(self.bn2(self.conv2(s16)))
        s8 = s8 + s16
        s8 = F.relu(self.bn3(self.conv3(s8)))

        # extract finger knuckle features output [b, 128, 44, 52]
        # input x: [b, c, 176, 208]
        # output x: [b, c, 44, 52]
        x = self.fknet(x)

        # fuse feature maps by concat
        # [b, 256, 32, 32]
        s8 = self.upsample(s8)
        out = torch.cat([x, s8], dim=1)
        out = F.relu(self.middlebn(self.middle(out)))
        # out = x + self.upsample(s8)
        out = self.resid1(out)
        out = self.resid2(out)
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.conv6(out))

        return out
