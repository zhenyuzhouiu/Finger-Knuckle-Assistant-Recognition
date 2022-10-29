import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import torch
import torchvision.utils
from torch import nn, optim
from models import loss_function
from models.net_common import ConvLayer
from models.EfficientNetV2 import EfficientVSResidual
import torch.nn.functional as F


class FirstSTNetThenConvNetMiddleEfficientNet(torch.nn.Module):
    def __init__(self):
        super(FirstSTNetThenConvNetMiddleEfficientNet, self).__init__()
        # Spatial Transformer Network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, bias=False),
            nn.BatchNorm2d(num_features=8),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(16 * 28 * 28, 28 * 28),
            nn.ReLU(True),
            nn.Linear(28 * 28, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[4].weight.data.zero_()
        self.fc_loc[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # Initial convolution layers
        # input shape:-> [bs, c, h ,w]:[bs, 3, 128, 128]
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(num_features=128)
        # output shape:-> [bs, 128, 32, 32]
        self.efficient = EfficientVSResidual(width_coefficient=1, depth_coefficient=1, drop_connect_rate=0.)
        self.conv4 = ConvLayer(128, 64, kernel_size=1, stride=1, bias=False)
        self.bn4 = torch.nn.BatchNorm2d(num_features=64)
        self.conv5 = ConvLayer(64, 1, kernel_size=1, stride=1)
        # self.bn5 = torch.nn.BatchNorm2d(num_features=1)

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 16 * 28 * 28)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def forward(self, x):
        stn_x = self.stn(x)
        conv1 = F.relu(self.bn1(self.conv1(stn_x)))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        conv3 = F.relu(self.bn3(self.conv3(conv2)))
        efficient = self.efficient(conv3)
        conv4 = F.relu(self.bn4(self.conv4(efficient)))
        conv5 = F.relu(self.conv5(conv4))

        return conv5


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(5, 5, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(5 * 28 * 28, 5 * 28),
            nn.ReLU(True),
            nn.Linear(5 * 28, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 5 * 28 * 28)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), )
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        return x


if __name__ == "__main__":
    model_path = "../checkpoint/fkv3-session2_Net-shiftedloss-lr0.01-subs8-angle5-a20-s4_2022-05-13-11-34/ckpt_epoch_3000.pth"
    data_path = "../dataset/PolyUKnuckleV3/Session_2_128/"
    subject_path = []
    for s in os.listdir(data_path):
        subject_path.append(os.path.join(data_path, s))

    model = Net().to(device=0)
    loss = loss_function.ShiftedLoss(3, 3).to(device=0)
    with torch.no_grad():
        model.eval()
        loss.eval()
        model.load_state_dict(torch.load(model_path))
        input_data = torch.zeros([60, 3, 128, 128], device=0)
        for test in range(20):
            bs = 0
            for i in range(0+10*test, 10+10*test):
                for j in os.listdir(subject_path[i]):
                    image_path = os.path.join(subject_path[i], j)
                    image = mpimage.imread(image_path)

                    image = torch.from_numpy(image / 255.).permute(2, 0, 1)
                    input_data[bs, :, :, :] = image
                    bs += 1

            in_grid = torchvision.utils.make_grid(input_data.cpu(), nrow=6).permute(1, 2, 0)
            transform_data = model.stn(input_data)
            out_grid = torchvision.utils.make_grid(transform_data.cpu(), nrow=6).permute(1, 2, 0)

            # plot the results side-by-side
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(in_grid)
            axarr[0].set_title("Dataset Images")
            axarr[1].imshow(out_grid)
            axarr[1].set_title("Transformer Images")

            plt.ioff()
            plt.show()
