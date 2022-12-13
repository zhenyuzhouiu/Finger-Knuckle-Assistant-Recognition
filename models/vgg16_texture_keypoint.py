import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from loss.sinkhorn import *


def featureL2Norm(feature, dim="channel"):
    """
    github: https://github.com/ignacio-rocco/cnngeometric_pytorch
    paper: Convolutional neural network architecture for geometric matching
    """
    # feature.shape:-> [b, c, h, w]
    b, c, h, w = feature.shape
    epsilon = 1e-6
    if dim == "channel":
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    else:
        norm = torch.pow(torch.sum(torch.pow(feature, 2).view(b, c, -1), -1) + epsilon, 0.5)
        norm = norm.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)

    return torch.div(feature, norm)


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """
    Perform Sinkhorn Normalization in Log-space for stability
    https://github.com/magicleap/SuperGluePretrainedNetwork
    """
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """
    Perform Differentiable Optimal Transport in Log-space for stability
    https://github.com/magicleap/SuperGluePretrainedNetwork
    """
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

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
                features32 = featureL2Norm(features32, dim='texture')
                features8 = featureL2Norm(features8, dim='channel')
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

    def __init__(self, shape='3D', normalization=True, matching_type='superglue', sinkhorn_it=100):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.matching_type = matching_type
        self.shape = shape
        self.ReLU = nn.ReLU()
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        self.sinkhorn_it = sinkhorn_it

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

            # from the "Convolutional neural network architecture for geometric matching"
            # correlation_tensor should be normalized for non-maximal suppressing
            # however, from the superglue, the correlation_tensor should not be normalized
            # for representing predicted confidence
            if self.normalization:
                correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))
            bs, ch, he, we = correlation_tensor.shape
            # transform the correlation_tensor to correlation_matrix
            # correlation_matrix.shape:-> [b, 64, 64]
            score_matrix = correlation_tensor.view(bs, ch, -1)
            score_matrix = score_matrix / (c ** 0.5)
            optimal_p = log_optimal_transport(score_matrix, self.bin_score,
                                              iters=self.sinkhorn_it)
            optimal_p = torch.exp(optimal_p)[:, :-1, :-1]
            # ot will be larger with two images are more similar
            ot = torch.sum(optimal_p.mul(score_matrix).view(b, -1), -1)

            return torch.exp(-ot)

        if self.matching_type == 'superglue':
            # feature_A.shape==feature_B.shape:-> [b, c, 8, 8]
            keypoint_A = feature_A.view(b, c, -1)  # shape:-> [b, c, 64]
            keypoint_B = feature_B.view(b, c, -1)
            correlation_tensor = torch.einsum('bdn,bdm->bnm', keypoint_A, keypoint_B)

            # from the "Convolutional neural network architecture for geometric matching"
            # correlation_tensor should be normalized for non-maximal suppressing
            # however, from the superglue, the correlation_tensor should not be normalized
            # for representing predicted confidence
            if self.normalization:
                correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))
            bs, n, m = correlation_tensor.shape
            # transform the correlation_tensor to correlation_matrix
            # correlation_matrix.shape:-> [b, 64, 64]
            score_matrix = correlation_tensor
            score_matrix = score_matrix / (c ** 0.5)
            optimal_p = log_optimal_transport(score_matrix, self.bin_score, iters=self.sinkhorn_it)
            optimal_p = torch.exp(optimal_p)[:, :-1, :-1]
            # ot will be larger with two images are more similar
            ot = torch.sum(optimal_p.mul(score_matrix).view(b, -1), -1)

            return torch.exp(-ot)

        if self.matching_type == 'subtraction':
            return feature_A.sub(feature_B)

        if self.matching_type == 'concatenation':
            return torch.cat((feature_A, feature_B), 1)


if __name__ == "__main__":
    image_batch = torch.arange(0, 128 * 128, dtype=torch.float).view(1, 1, 128, 128).repeat(1, 3, 1, 1).cuda()
    image_batch = Variable(image_batch, requires_grad=False)
    fe = FeatureExtraction().eval()
    features = fe(image_batch)
