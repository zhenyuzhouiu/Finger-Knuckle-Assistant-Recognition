# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
import warnings

import torch
import math
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm
from data.augmentations import resize_img
from torch.optim.lr_scheduler import MultiStepLR
from models.superglue_gnn import SuperGlue


def generate_theta(i_radian, i_tx, i_ty, i_batch_size, i_h, i_w, i_dtype):
    # if you want to keep ration when rotation a rectangle image
    theta = torch.tensor([[math.cos(i_radian), math.sin(-i_radian) * i_h / i_w, i_tx],
                          [math.sin(i_radian) * i_w / i_h, math.cos(i_radian), i_ty]],
                         dtype=i_dtype).unsqueeze(0).repeat(i_batch_size, 1, 1)
    # else
    # theta = torch.tensor([[math.cos(i_radian), math.sin(-i_radian), i_tx],
    #                       [math.sin(i_radian), math.cos(i_radian), i_ty]],
    #                      dtype=i_dtype).unsqueeze(0).repeat(i_batch_size, 1, 1)
    return theta


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    #
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        win (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):
    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)
    # win = Variable(win, requires_grad=False)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    # cs_map.shape = ssim_map.shape:-> [b, c, h, w]
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    # torch.flatten(input, start_dim, end_dim)
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(
        X,
        Y,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        win=None,
        K=(0.01, 0.03),
        nonnegative_ssim=False,
):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    # squeeze dimension 2 & 3
    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        # win.shape:-> [1, 1, win_size]
        win = _fspecial_gauss_1d(win_size, win_sigma)
        # win.shape:-> [1, 1, 1, win_size]
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    # ssim_per_channel.shape = cs.shape:-> [b, c]
    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
        X, Y, data_range=255, size_average=True, win_size=11, win_sigma=1.5, win=None, weights=None, K=(0.01, 0.03)
):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
            2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = X.new_tensor(weights)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIM(torch.nn.Module):
    def __init__(
            self,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            channel=3,
            spatial_dims=2,
            K=(0.01, 0.03),
            nonnegative_ssim=False,
    ):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        # self.win.shape:-> [channel, 1, 1, win_size]
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return 1 - ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class RSSSIM(torch.nn.Module):
    def __init__(
            self,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            channel=3,
            spatial_dims=2,
            K=(0.01, 0.03),
            nonnegative_ssim=False,
            v_shift=0,
            h_shift=0,
            angle=0
    ):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(RSSSIM, self).__init__()
        self.win_size = win_size
        # self.win.shape:-> [channel, 1, 1, win_size]
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim
        self.v_shift = v_shift
        self.h_shift = h_shift
        self.angle = angle

    def forward(self, X, Y):
        b, c, h, w = X.shape
        mask = torch.ones_like(Y, device=Y.device)
        n_affine = 0
        min_dist = torch.zeros([b, ], dtype=X.dtype, device=X.device)
        # if self.training:
        #     min_dist = torch.zeros([b, ], dtype=X.dtype, requires_grad=True, device=X.device)
        # else:
        #     min_dist = torch.zeros([b, ], dtype=X.dtype, requires_grad=False, device=X.device)

        if self.v_shift == self.h_shift == self.angle == 0:
            min_dist = 1 - ssim(X, Y, data_range=self.data_range, size_average=self.size_average,
                                win=self.win, K=self.K, nonnegative_ssim=self.nonnegative_ssim)
            return min_dist
        for tx in range(-self.h_shift, self.h_shift + 1):
            for ty in range(-self.v_shift, self.v_shift + 1):
                for a in range(-self.angle, self.angle + 1):
                    radian_a = a * math.pi / 180.
                    ratio_tx = 2 * tx / w
                    ratio_ty = 2 * ty / h
                    theta = generate_theta(radian_a, ratio_tx, ratio_ty, b, h, w, Y.dtype).to(Y.device)
                    grid = F.affine_grid(theta, Y.size(), align_corners=False).to(Y.device)
                    r_Y = F.grid_sample(Y, grid, align_corners=False)
                    # mean_se.shape: -> (bs, )
                    mean_ssim = 1 - ssim(X, r_Y, data_range=self.data_range, size_average=self.size_average,
                                         win=self.win, K=self.K, nonnegative_ssim=self.nonnegative_ssim)
                    if n_affine == 0:
                        min_dist = mean_ssim
                    else:
                        min_dist = torch.vstack([min_dist, mean_ssim])
                        # min_dist, _ = torch.min(min_dist, dim=0)
                    n_affine += 1

        min_dist, _ = torch.min(min_dist, dim=0)
        return min_dist


class SSIMGNN(torch.nn.Module):
    def __init__(
            self,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            channel=3,
            spatial_dims=2,
            K=(0.01, 0.03),
            nonnegative_ssim=False,
            config={}
    ):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIMGNN, self).__init__()
        self.gnn = SuperGlue(config=config)
        self.win_size = win_size
        # self.win.shape:-> [channel, 1, 1, win_size]
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, X_mask, Y, Y_mask):
        X, Y = self.gnn(X, Y)
        return 1 - ssim(
            X,
            X_mask,
            Y,
            Y_mask,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class MS_SSIM(torch.nn.Module):
    def __init__(
            self,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            channel=3,
            spatial_dims=2,
            weights=None,
            K=(0.01, 0.03),
    ):
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        )


if __name__ == "__main__":
    src_image = cv2.imread("/home/zhenyuzhou/Pictures/timberlake.jpg")
    src_image = resize_img(src_image, size=(1080, 1440))
    src_image = transforms.ToTensor()(src_image).to("cuda").unsqueeze(0)

    src2_image = cv2.imread("/home/zhenyuzhou/Pictures/zhenyuzhou.jpg")
    src2_image = resize_img(src2_image, size=(1080, 1440))
    src2_image = transforms.ToTensor()(src2_image).to("cuda").unsqueeze(0)

    dst_image = torch.rand(src_image.shape, dtype=src_image.dtype).to("cuda")

    s_score = ssim(src_image, X_mask=0, Y=dst_image, Y_maks=0, data_range=1.0)
    print("The beginning similarity score: " + str(s_score))

    src_image = Variable(src_image, requires_grad=False)
    dst_image = Variable(dst_image, requires_grad=True)

    optim = Adam(params=[dst_image], lr=0.1)
    loss = SSIM(data_range=1.0, channel=3)
    pbar = tqdm(range(3000))
    scheduler = MultiStepLR(optim, milestones=[10, 500, 1000], gamma=0.1)
    for epoch in pbar:
        optim.zero_grad()
        ls = (loss(src_image, 0, dst_image, 0) + loss(src2_image, 0, dst_image, 0))
        ls.backward()
        optim.step()
        pbar.set_description(f'Epoch [{epoch}/{3000}]')
        pbar.set_postfix(loss_inference="{:.6f}".format(ls.item()))

        scheduler.step()

    s_image = src_image.squeeze(0).data.cpu().numpy().transpose(1, 2, 0)
    s_image = cv2.cvtColor(s_image, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 3, 1)
    plt.title("source image")
    plt.imshow(s_image)
    s2_image = src2_image.squeeze(0).data.cpu().numpy().transpose(1, 2, 0)
    s2_image = cv2.cvtColor(s2_image, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 3, 2)
    plt.title("source2 image")
    plt.imshow(s2_image)
    d_image = dst_image.squeeze(0).data.cpu().numpy().transpose(1, 2, 0)
    d_image = cv2.cvtColor(d_image, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 3, 3)
    plt.title("learnable image")
    plt.imshow(d_image)
    plt.show()
