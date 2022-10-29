import torch
from torch import nn

class Bottlenect(nn.Module):
    """
    Block的各个plane值：
        inplane：输出block的之前的通道数
        midplane：在block中间处理的时候的通道数（这个值是输出维度的1/4）
        midplane*self.extention：输出的维度

            # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    """
    def __init__(self, inplane, midplane, stride, downsample=None):
        super(Bottlenect, self).__init__()
        self.extention=4

        self.conv1 = nn.Conv2d(in_channels=inplane, out_channels=midplane, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(midplane)
        self.conv2 = nn.Conv2d(midplane, midplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(midplane)
        self.conv3 = nn.Conv2d(midplane, midplane*self.extention, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(midplane*self.extention)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplane = 64

        self.block = block
        self.layers = layers

        # stem
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        # 64, 128, 256, 512是指扩大4倍之前的维度
        self.stage1 = self.make_layer(self.block, 64, self.layers[0], stride=1)
        self.stage2 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.stage3 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        self.stage4 = self.make_layer(self.block, 512, self.layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512*block.extenstion, num_classes)

    def make_layer(self, block, midplane, block_num, stride=1):
        """
            block:block模块
            midplane：每个模块中间运算的维度，一般等于输出维度/4
            block_num：重复次数
            stride：Conv Block的步长
        """

        block_list = []

        # 先计算要不要加downsample模块
        downsample = None
        if stride != 1 or self.inplane != midplane*block.extenstion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplane, midplane*block.extenstion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(midplane*block.extenstion)
            )

        # Conv Block
        conv_block = block(self.inplane, midplane, stride=stride, downsample=downsample)
        block_list.append(conv_block)
        self.inplane = midplane*block.extenstion

        # Identity Block
        for i in range(1, block_num):
            block_list.append(block(self.inplane, midplane, stride=1))

        return nn.Sequential(*block_list)

    def forward(self, x):
        # stem部分: conv+bn+relu+maxpool
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # block
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        # 分类
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out



class FKNet(nn.Module):
    def __init__(self, block, layers, num_classes=190):
        super(ResNet, self).__init__()
        self.inplane = 64

        self.block = block
        self.layers = layers

        # stem
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        # 64, 128, 256, 512是指扩大4倍之前的维度
        self.stage1 = self.make_layer(self.block, 64, self.layers[0], stride=1)
        self.stage2 = self.make_layer(self.block, 128, self.layers[1], stride=2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128*block.extenstion, out_channels=190, kernel_size=[6,10])

    def make_layer(self, block, midplane, block_num, stride=1):
        """
            block:block模块
            midplane：每个模块中间运算的维度，一般等于输出维度/4
            block_num：重复次数
            stride：Conv Block的步长
        """

        block_list = []

        # 先计算要不要加downsample模块
        downsample = None
        if stride != 1 or self.inplane != midplane*block.extenstion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplane, midplane*block.extenstion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(midplane*block.extenstion)
            )

        # Conv Block
        conv_block = block(self.inplane, midplane, stride=stride, downsample=downsample)
        block_list.append(conv_block)
        self.inplane = midplane*block.extenstion

        # Identity Block
        for i in range(1, block_num):
            block_list.append(block(self.inplane, midplane, stride=1))

        return nn.Sequential(*block_list)

    def forward(self, x):
        # stem部分: conv+bn+relu+maxpool
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out1 = self.maxpool(out)

        # block
        out2 = self.stage1(out1)

        out3 = self.stage2(out2)

        out4 = torch.cat([out3, self.pool(out2), self.pool(out1)])

        out5 = self.conv4(out4)
        return out5