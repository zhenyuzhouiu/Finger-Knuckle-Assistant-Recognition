import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from loss.sinkhorn import *


def featureL2Norm(feature):
    """
    github: https://github.com/ignacio-rocco/cnngeometric_pytorch
    paper: Convolutional neural network architecture for geometric matching
    """
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)


class FeatureExtraction(torch.nn.Module):
    """
    github: https://github.com/ignacio-rocco/cnngeometric_pytorch
    paper: Convolutional neural network architecture for geometric matching
    """

    def __init__(self, train_fe=False, feature_extraction_cnn='vgg', normalization=True,
                 last_layer=['relu3_3', 'relu5_3'], use_cuda=True):
        super(FeatureExtraction, self).__init__()
        self.normalization = normalization
        if feature_extraction_cnn == 'vgg':
            self.model = models.vgg16(pretrained=True)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
                                  'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
                                  'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
                                  'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                                  'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5']
            if last_layer == '':
                last_layer = 'pool4'
                last_layer_idx = vgg_feature_layers.index(last_layer)
                self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx + 1])
            elif len(last_layer) == 1:
                last_layer_idx = vgg_feature_layers.index(last_layer)
                self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx + 1])
            else:
                if len(last_layer) != 2:
                    raise RuntimeError("Please make sure 'last_layer' is a correct input")
                self.two = True
                index32 = vgg_feature_layers.index(last_layer[0])
                self.feature32 = nn.Sequential(*list(self.model.features.children())[:index32 + 1])
                index8 = vgg_feature_layers.index(last_layer[1])
                self.feature8 = nn.Sequential(*list(self.model.features.children())[index32 + 1:index8 + 1])

        if feature_extraction_cnn == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer == '':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]

            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx + 1])
        if feature_extraction_cnn == 'resnet101_v2':
            self.model = models.resnet101(pretrained=True)
            # keep feature extraction network up to pool4 (last layer - 7)
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if feature_extraction_cnn == 'densenet201':
            self.model = models.densenet201(pretrained=True)
            # keep feature extraction network up to denseblock3
            # self.model = nn.Sequential(*list(self.model.features.children())[:-3])
            # keep feature extraction network up to transitionlayer2
            self.model = nn.Sequential(*list(self.model.features.children())[:-4])
        if not train_fe:
            # freeze parameters
            if self.two:
                for param in self.feature32.parameters():
                    param.requires_grad = False
                for param in self.feature8.parameters():
                    param.requires_grad = False
            else:
                for param in self.model.parameters():
                    param.requires_grad = False
        # move to GPU
        if use_cuda:
            self.model = self.model.cuda()

    def forward(self, image_batch):
        if self.two:
            features32 = self.feature32(image_batch)
            features8 = self.feature8(features32)
            if self.normalization:
                features32 = featureL2Norm(features32)
                features8 = featureL2Norm(features8)
            return features32, features8
        else:
            features = self.model(image_batch)
            if self.normalization:
                features = featureL2Norm(features)
            return features


class FeatureCorrelation(torch.nn.Module):
    """
    github: https://github.com/ignacio-rocco/cnngeometric_pytorch
    paper: Convolutional neural network architecture for geometric matching
    """

    def __init__(self, shape='3D', normalization=True, matching_type='correlation'):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.matching_type = matching_type
        self.shape = shape
        self.ReLU = nn.ReLU()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        if self.matching_type == 'correlation':
            if self.shape == '3D':
                # reshape features for matrix multiplication
                feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
                feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
                # perform matrix mult.
                feature_mul = torch.bmm(feature_B, feature_A)
                # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
                correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
            elif self.shape == '4D':
                # reshape features for matrix multiplication
                feature_A = feature_A.view(b, c, h * w).transpose(1, 2)  # size [b,c,h*w]
                feature_B = feature_B.view(b, c, h * w)  # size [b,c,h*w]
                # perform matrix mult.
                feature_mul = torch.bmm(feature_A, feature_B)
                # indexed [batch,row_A,col_A,row_B,col_B]
                correlation_tensor = feature_mul.view(b, h, w, h, w).unsqueeze(1)

            if self.normalization:
                correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))
            bs, ch, he, we = correlation_tensor.shape
            # transform the correlation_tensor to correlation_matrix
            # correlation_matrix.shape:-> [b, 64, 64]
            correlation_matrix = correlation_tensor.view(bs, ch, -1)





            return 0

        if self.matching_type == 'subtraction':
            return feature_A.sub(feature_B)

        if self.matching_type == 'concatenation':
            return torch.cat((feature_A, feature_B), 1)


if __name__ == "__main__":
    image_batch = torch.arange(0, 128 * 128, dtype=torch.float).view(1, 1, 128, 128).repeat(1, 3, 1, 1).cuda()
    image_batch = Variable(image_batch, requires_grad=False)
    fe = FeatureExtraction().eval()
    features = fe(image_batch)