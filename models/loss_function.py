import math
import sys
import numpy
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.gradcheck import gradcheck

torch.backends.cudnn.deterministic = True


# The whole image rotation and translation loss cannot back propagation by cv2;
# But it can back propagation by affine_grid and grid_sample

######################
# Gradient Check True
# WholeImageRotationAndTranslation
# ShiftedLoss
# ImageBlockRotationAndTranslation
######################
# Gradient Check False
#
######################


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


def rotate_mse_loss(i_fm1, i_mask1, i_fm2, i_mask2):
    """
    fm:-> feature maps
    mask:-> mark geometric transformations
    shape:-> [bs, 1, h, w] for the RFN, h=w=32
    """
    # common region
    mask = torch.mul(i_mask1, i_mask2)
    # only calculate pixel-to-pixel MSE on the common region
    square_err = torch.mul(torch.pow((i_fm1 - i_fm2), 2), mask)
    mean_se = square_err.view(i_fm1.size(0), -1).sum(1) / (mask.view(i_fm1.size(0), -1).sum(1) + 0.00000000001)
    return mean_se


def mse_loss(src, target):
    if isinstance(src, torch.autograd.Variable):
        return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.data.nelement() * src.size(0)
    else:
        return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.nelement() * src.size(0)


class CosineSimilarity(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def forward(self, i_fm1, i_fm2):
        # b, 1, 32, 32
        b, c, h, w = i_fm1.shape
        i_fm1 = i_fm1.contiguous().view(b, -1)
        i_fm2 = i_fm2.contiguous().view(b, -1)
        cosim = torch.cosine_similarity(i_fm1, i_fm2, dim=1)

        return cosim


class RSIL(torch.nn.Module):
    """
    Rotation and Shift Invariant Loss Function
    The grid_sample is nondeterministic when using CUDA backend
    So I recommend to use CPU
    """

    def __init__(self, i_v_shift, i_h_shift, i_angle):
        super(RSIL, self).__init__()
        self.v_shift = i_v_shift
        self.h_shift = i_h_shift
        self.angle = i_angle

    def forward(self, i_fm1, i_fm2, i_min_or_max="min"):
        b, c, h, w = i_fm1.shape
        mask = torch.ones_like(i_fm2, device=i_fm1.device)
        n_affine = 0
        if self.training:
            min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=True, device=i_fm1.device)
        else:
            min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=False, device=i_fm1.device)

        if self.v_shift == self.h_shift == self.angle == 0:
            min_dist = mse_loss(i_fm1, i_fm2).cuda()
            return min_dist
        for tx in range(-self.h_shift, self.h_shift + 1):
            for ty in range(-self.v_shift, self.v_shift + 1):
                for a in range(-self.angle, self.angle + 1):
                    radian_a = a * math.pi / 180.
                    ratio_tx = 2 * tx / w
                    ratio_ty = 2 * ty / h
                    theta = generate_theta(radian_a, ratio_tx, ratio_ty, b, h, w, i_fm1.dtype).to(i_fm1.device)
                    grid = F.affine_grid(theta, i_fm2.size(), align_corners=False).to(i_fm1.device)
                    r_fm2 = F.grid_sample(i_fm2, grid, align_corners=False)
                    r_mask = F.grid_sample(mask, grid, align_corners=False)
                    # mean_se.shape: -> (bs, )
                    mean_se = rotate_mse_loss(i_fm1, r_fm2, r_mask)
                    if n_affine == 0:
                        min_dist = mean_se
                    else:
                        min_dist = torch.vstack([min_dist, mean_se])
                    n_affine += 1

        if i_min_or_max == "min":
            min_dist, _ = torch.min(min_dist, dim=0)
        elif i_min_or_max == "max":
            min_dist, _ = torch.max(min_dist, dim=0)
        else:
            raise RuntimeError(
                "Please make sure the RSIL loss function with right i_min_or_max")

        return min_dist


class MaskRSIL(torch.nn.Module):
    """
    initial input:
    =============== i_v_shift, i_h_shift, and i_angle should be even numbers
    forward input:->
    =============== i_fm1, i_mask1, i_fm2, i_mask2
    """
    def __init__(self, i_v_shift, i_h_shift, i_angle):
        super(MaskRSIL, self).__init__()
        self.v_shift = i_v_shift
        self.h_shift = i_h_shift
        self.angle = i_angle

    def forward(self, i_fm1, i_mask1, i_fm2, i_mask2, i_min_or_max="min"):
        b, c, h, w = i_fm1.shape
        n_affine = 0
        if self.training:
            min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=True, device=i_fm1.device)
        else:
            min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=False, device=i_fm1.device)

        if self.v_shift == self.h_shift == self.angle == 0:
            # min_dist = mse_loss(i_fm1, i_fm2).cuda()
            min_dist = rotate_mse_loss(i_fm1, i_mask1, i_fm2, i_mask2)
            return min_dist
        for tx in range(-self.h_shift, self.h_shift + 1):
            for ty in range(-self.v_shift, self.v_shift + 1):
                for a in range(-self.angle, self.angle + 1):
                    # transform i_fm1
                    # radian_a = -(a * math.pi / 180.)
                    # ratio_tx = -(2 * tx / w)
                    # ratio_ty = -(2 * ty / h)
                    # theta = generate_theta(radian_a, ratio_tx, ratio_ty, b, h, w, i_fm1.dtype).to(i_fm1.device)
                    # grid = F.affine_grid(theta, i_fm1.size(), align_corners=False).to(i_fm1.device)
                    # r_fm1 = F.grid_sample(i_fm1, grid, align_corners=False)
                    # r_mask1 = F.grid_sample(i_mask1, grid, align_corners=False)
                    # transform i_fm2
                    radian_a = a * math.pi / 180.
                    ratio_tx = 2 * tx / w
                    ratio_ty = 2 * ty / h
                    theta = generate_theta(radian_a, ratio_tx, ratio_ty, b, h, w, i_fm2.dtype).to(i_fm2.device)
                    grid = F.affine_grid(theta, i_fm2.size(), align_corners=False).to(i_fm2.device)
                    r_fm2 = F.grid_sample(i_fm2, grid, align_corners=False)
                    r_mask2 = F.grid_sample(i_mask2, grid, align_corners=False)
                    # mean_se.shape: -> (bs, )
                    mean_se = rotate_mse_loss(i_fm1, i_mask1, r_fm2, r_mask2)
                    if n_affine == 0:
                        min_dist = mean_se
                    else:
                        min_dist = torch.vstack([min_dist, mean_se])
                    n_affine += 1

        if i_min_or_max == "min":
            min_dist, _ = torch.min(min_dist, dim=0)
        elif i_min_or_max == "max":
            min_dist, _ = torch.max(min_dist, dim=0)
        else:
            raise RuntimeError(
                "Please make sure the RSIL loss function with right i_min_or_max")

        return min_dist


class ImageBlockRotationAndTranslation(torch.nn.Module):
    def __init__(self, i_block_size, i_v_shift, i_h_shift, i_angle, i_topk=16):
        super(ImageBlockRotationAndTranslation, self).__init__()
        self.block_size = i_block_size
        self.v_shift = i_v_shift
        self.h_shift = i_h_shift
        self.angle = i_angle
        self.topk = i_topk

    def forward(self, i_fm1, i_fm2):
        b, c, h, w = i_fm1.shape
        if self.training:
            min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=True, device=i_fm1.device)
        else:
            min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=False, device=i_fm1.device)

        if self.v_shift == self.h_shift == self.angle == 0:
            min_dist = mse_loss(i_fm1, i_fm2).cuda()
            return min_dist

        n_affine = 0
        for sub_x in range(0, w, self.block_size):
            for sub_y in range(0, h, self.block_size):
                sub_fm1 = i_fm1[:, :, sub_y:sub_y + self.block_size, sub_x:sub_x + self.block_size]

                sub_affine = 0
                if self.training:
                    sub_min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=True, device=i_fm1.device)
                else:
                    sub_min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=False, device=i_fm1.device)

                for dx in range(-self.h_shift, self.h_shift + 1):
                    for dy in range(-self.v_shift, self.v_shift + 1):
                        if sub_y + dy < 0:
                            if sub_x + dx < 0:
                                sub_fm2 = i_fm2[:, :, 0:self.block_size, 0:self.block_size]
                            elif sub_x + dx + self.block_size > w:
                                sub_fm2 = i_fm2[:, :, 0:self.block_size, w - self.block_size:w]
                            else:
                                sub_fm2 = i_fm2[:, :, 0:self.block_size, sub_x + dx:sub_x + self.block_size + dx]
                        elif sub_y + dy + self.block_size > h:
                            if sub_x + dx < 0:
                                sub_fm2 = i_fm2[:, :, h - self.block_size:h, 0:self.block_size]
                            elif sub_x + dx + self.block_size > w:
                                sub_fm2 = i_fm2[:, :, h - self.block_size:h,
                                          w - self.block_size:w]
                            else:
                                sub_fm2 = i_fm2[:, :, h - self.block_size:h,
                                          sub_x + dx:sub_x + self.block_size + dx]
                        else:
                            if sub_x + dx < 0:
                                sub_fm2 = i_fm2[:, :, sub_y + dy:sub_y + self.block_size + dy, 0:self.block_size]
                            elif sub_x + dx + self.block_size > w:
                                sub_fm2 = i_fm2[:, :, sub_y + dy:sub_y + self.block_size + dy,
                                          w - self.block_size:w]
                            else:
                                sub_fm2 = i_fm2[:, :, sub_y + dy:sub_y + self.block_size + dy,
                                          sub_x + dx:sub_x + self.block_size + dx]

                        for a in range(-self.angle, self.angle + 1):
                            sub_fm2_b, sub_fm2_c, sub_fm2_h, sub_fm2_w = sub_fm2.shape
                            mask = torch.ones_like(sub_fm2, dtype=i_fm1.dtype, device=i_fm1.device)
                            radian_a = a * math.pi / 180.
                            theta = generate_theta(radian_a, 0, 0, sub_fm2_b, sub_fm2_h, sub_fm2_w, i_fm1.dtype)
                            grid = F.affine_grid(theta, sub_fm2.size(), align_corners=False.device)
                            r_sub_fm2 = F.grid_sample(sub_fm2, grid, align_corners=False)
                            r_mask = F.grid_sample(mask, grid, align_corners=False)
                            sub_mean_se = rotate_mse_loss(sub_fm1, r_sub_fm2, r_mask)
                            if sub_affine == 0:
                                sub_min_dist = sub_mean_se
                            else:
                                sub_min_dist = torch.vstack([sub_min_dist, sub_mean_se])
                            sub_affine += 1

                sub_min_dist, _ = torch.min(sub_min_dist, dim=0)
                if n_affine == 0:
                    min_dist = sub_min_dist
                else:
                    min_dist = torch.vstack([min_dist, sub_min_dist])
                n_affine += 1

        if self.training:
            min_dist = torch.sum(min_dist, dim=0)
        else:
            min_dist, _ = torch.topk(min_dist, self.topk, dim=0, largest=False)
            min_dist = torch.sum(min_dist, dim=0)

        return min_dist


class ShiftedLoss(torch.nn.Module):
    def __init__(self, hshift, vshift):
        super(ShiftedLoss, self).__init__()
        self.hshift = hshift
        self.vshift = vshift

    def forward(self, fm1, fm2):
        # C * H * W
        bs, _, h, w = fm1.size()
        if w < self.hshift:
            self.hshift = self.hshift % w
        if h < self.vshift:
            self.vshift = self.vshift % h

        # min_dist shape (bs, )  & sys.float_info.max is to get the max float value
        # min_dist save the maximal float value
        min_dist = torch.ones(bs).cuda() * sys.float_info.max
        # torch is set tensor as an instance of Variable
        if isinstance(fm1, torch.autograd.Variable):
            min_dist = Variable(min_dist, requires_grad=False)

        if self.hshift == 0 and self.vshift == 0:
            dist = mse_loss(fm1, fm2).cuda()
            min_dist, _ = torch.min(torch.stack([min_dist, dist]), 0)
            return min_dist

        for bh in range(-self.hshift, self.hshift + 1):
            for bv in range(-self.vshift, self.vshift + 1):
                if bh >= 0:
                    ref1, ref2 = fm1[:, :, :, :w - bh], fm2[:, :, :, bh:]
                else:
                    ref1, ref2 = fm1[:, :, :, -bh:], fm2[:, :, :, :w + bh]

                if bv >= 0:
                    ref1, ref2 = ref1[:, :, :h - bv, :], ref2[:, :, bv:, :]
                else:
                    ref1, ref2 = ref1[:, :, -bv:, :], ref2[:, :, :h + bv, :]
                dist = mse_loss(ref1, ref2).cuda()
                min_dist, _ = torch.min(torch.stack([min_dist.squeeze(), dist.squeeze()]), 0)
        return min_dist.squeeze()


class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, i_fm1, i_fm2):
        square_err = ((i_fm1 - i_fm2) ** 2).view(i_fm1.size(0), -1).sum(1)
        num = i_fm1.view(i_fm1.size(0), -1).size(1)
        mse = square_err / num
        return mse


class HammingDistance(torch.nn.Module):
    def __init__(self):
        super(HammingDistance, self).__init__()

    def forward(self, i_fm1, i_fm2):
        xor = torch.abs(torch.sub(i_fm1, i_fm2))
        hamming_distance = (xor.view(i_fm1.size(0), -1)).sum(1)
        return hamming_distance


class ConvolutionScore(torch.nn.Module):
    """
    Split one of feature maps from [bs, 1, 32, 32] to [bs, 16, 8, 8] as the convolution kernel to
    convolve the other [bs, 1, 32, 32], then output [bs, 16, 32, 32] as the attention matching score.
    1:-> input feature maps' shape is [bs, 1, 32, 32]
    2:-> split feature maps to 8x8 sub-images,
    3:-> one sub-image compare to the another feature maps to calculate the similarity
    4:-> generate [bs, 4x4, 32, 32] channels attention score
         the attention score is bigger when two feature maps are more similarity

    """

    def __init__(self, i_subsize=8):
        super(ConvolutionScore, self).__init__()
        self.subsize = i_subsize
        self.reflection_pad = torch.nn.ReflectionPad2d(3)

    def forward(self, i_fm1, i_fm2):
        # torch.cat, torch.stack, torch.chunk, torch.split
        # view operator cannot work, because it will firstly reshape the tensor with row
        # feature_kernel = i_fm1.view(bs, -1, 8, 8)
        # permute operator is just change dimension sequence
        bs, _, h, w = i_fm1.shape
        reflection_fm2 = self.reflection_pad(i_fm2)
        attention_score = torch.zeros([bs, ], dtype=i_fm1.dtype, device=i_fm1.device)
        for b in range(bs):
            feature_kernel = torch.zeros([16, 8, 8], dtype=i_fm1.dtype, device=i_fm1.device)
            for i in range(4):
                for j in range(4):
                    feature_kernel[i * 4 + j, :, :] = i_fm1[b, :, 8 * i:8 * (1 + i), 8 * j:8 * (1 + j)].squeeze()
            # attention shape:-> [1, 16, 16, 16]
            feature_kernel = feature_kernel.unsqueeze(1)
            attention = torch.nn.functional.conv2d(input=reflection_fm2[b, :, :, :].unsqueeze(0), weight=feature_kernel,
                                                   bias=None, stride=2, padding=0)
            # attention_similarity:-> [1, 1, 16, 16]
            attention_similarity, _ = torch.max(attention, dim=1)
            # attention_score:-> [1, 1]
            attention_mean = torch.mean(attention_similarity)
            # attention_score[b,] = torch.exp(-attention_mean)
            attention_score[b,] = 1 / (attention_mean + 1e-24)

        return attention_score


class WholeRotationShiftedLoss(torch.nn.Module):
    def __init__(self, hshift, vshift, angle):
        super(WholeRotationShiftedLoss, self).__init__()
        self.hshift = hshift
        self.vshift = vshift
        self.angle = angle

    def rotate_mse_loss(self, src, target, mask):
        # if isinstance(src, torch.autograd.Variable):
        #     return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.data.nelement() * src.size(0)
        # else:
        #     return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.nelement() * src.size(0)
        se = (src - target) ** 2
        mask_se = se * mask
        sum_se = mask_se.view(src.size(0), -1).sum(1)
        sum = mask.view(src.size(0), -1).sum(1)
        mse = sum_se / sum
        return mse

    def mse_loss(self, src, target):
        if isinstance(src, torch.autograd.Variable):
            return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.data.nelement() * src.size(0)
        else:
            return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.nelement() * src.size(0)

    def forward(self, fm1, fm2):
        # C * H * W
        bs, _, h, w = fm1.size()
        if w < self.hshift:
            self.hshift = self.hshift % w
        if h < self.vshift:
            self.vshift = self.vshift % h

        # min_dist shape (bs, )  & sys.float_info.max is to get the max float value
        # min_dist save the maximal float value
        min_dist = torch.ones(bs, device=fm1.device) * sys.float_info.max
        # if isinstance(fm1, torch.autograd.Variable):
        #     min_dist = Variable(min_dist, requires_grad=True)
        if fm1.requires_grad:
            min_dist = Variable(min_dist, requires_grad=True)
        else:
            min_dist = Variable(min_dist, requires_grad=False)

        if self.hshift == 0 and self.vshift == 0:
            dist = self.mse_loss(fm1, fm2).to(fm1.device)
            min_dist, _ = torch.min(torch.stack([min_dist, dist]), 0)
            return min_dist

        for bh in range(-self.hshift, self.hshift + 1):
            for bv in range(-self.vshift, self.vshift + 1):
                if bh >= 0:
                    ref1, ref2 = fm1[:, :, :, :w - bh], fm2[:, :, :, bh:]
                else:
                    ref1, ref2 = fm1[:, :, :, -bh:], fm2[:, :, :, :w + bh]

                if bv >= 0:
                    ref1, ref2 = ref1[:, :, :h - bv, :], ref2[:, :, bv:, :]
                else:
                    ref1, ref2 = ref1[:, :, -bv:, :], ref2[:, :, :h + bv, :]

                for theta in range(-self.angle, self.angle + 1):
                    overlap_bs, overlap_c, overlap_h, overlap_w = ref1.size()
                    M = cv2.getRotationMatrix2D(center=(overlap_w / 2, overlap_h / 2), angle=theta, scale=1)
                    ref2 = torch.squeeze(ref2, dim=1)
                    n_ref2 = ref2.detach().cpu().numpy()
                    n_ref2 = n_ref2.transpose(1, 2, 0)
                    if overlap_bs > 512:
                        num_chuncks = overlap_bs // 512
                        num_reminder = overlap_bs % 512
                        r_ref2 = np.zeros(n_ref2.shape)
                        for nc in range(num_chuncks):
                            nc_ref2 = n_ref2[:, :, 0 + nc * 512:512 + nc * 512]
                            r_nc_ref2 = cv2.warpAffine(nc_ref2, M=M, dsize=(overlap_w, overlap_h))
                            r_ref2[:, :, 0 + nc * 512:512 + nc * 512] = r_nc_ref2
                        if num_reminder > 0:
                            nc_ref2 = n_ref2[:, :, 512 + nc * 512:]
                            r_nc_ref2 = cv2.warpAffine(nc_ref2, M=M, dsize=(overlap_w, overlap_h))
                            if r_nc_ref2.ndim == 2:
                                r_nc_ref2 = numpy.expand_dims(r_nc_ref2, axis=-1)
                            r_ref2[:, :, 512 + nc * 512:] = r_nc_ref2
                    else:
                        r_ref2 = cv2.warpAffine(n_ref2, M=M, dsize=(overlap_w, overlap_h))
                    # r_ref2 = rotate(n_ref2, angle=theta, reshape=False)

                    if r_ref2.ndim == 2:
                        r_ref2 = numpy.expand_dims(r_ref2, axis=-1)
                    r_ref2 = torch.from_numpy(r_ref2).to(fm1.device)
                    r_ref2 = r_ref2.permute(2, 0, 1).unsqueeze(1)

                    mask = np.ones([overlap_h, overlap_w])
                    r_mask = cv2.warpAffine(mask, M=M, dsize=(overlap_w, overlap_h))
                    # r_mask = rotate(mask, angle=theta, reshape=False)
                    r_mask = torch.from_numpy(r_mask).to(fm1.device)
                    r_mask = r_mask.unsqueeze(0).unsqueeze(0).repeat(overlap_bs, overlap_c, 1, 1)

                    dist = self.rotate_mse_loss(ref1, r_ref2, r_mask).to(fm1.device)

                    min_dist, _ = torch.min(torch.stack([min_dist.squeeze(), dist.squeeze()]), 0)
        return min_dist.squeeze()


if __name__ == "__main__":

    mode = "train"
    loss = "AttentionScore"
    if loss == "AttentionScore":
        if mode == "eval":
            loss = ConvolutionScore().eval()
            input1 = torch.randn([5, 5], dtype=torch.double, requires_grad=False).unsqueeze(0).unsqueeze(0)
            input2 = torch.randn([5, 5], dtype=torch.double, requires_grad=False).unsqueeze(0).unsqueeze(0)
            min_dist = loss(input1, input2)
            print(min_dist)
        else:
            loss = ConvolutionScore().train()
            input1 = torch.randn([32, 32], dtype=torch.double, requires_grad=True).unsqueeze(0).unsqueeze(0).repeat(2,
                                                                                                                    1,
                                                                                                                    1,
                                                                                                                    1)
            input2 = torch.randn([32, 32], dtype=torch.double, requires_grad=True).unsqueeze(0).unsqueeze(0).repeat(2,
                                                                                                                    1,
                                                                                                                    1,
                                                                                                                    1)
            min_dist = loss(input1, input2)
            print(min_dist)
            test = gradcheck(loss, [input1, input2])
            print("Are the gradients of whole image rotation and translation with undeformable correct: ", test)
    else:
        if mode == "eval":
            loss = RSIL(i_v_shift=3, i_h_shift=3, i_angle=3).eval()
            input1 = torch.randn([5, 5], dtype=torch.double, requires_grad=False).unsqueeze(0).unsqueeze(0)
            input2 = torch.randn([5, 5], dtype=torch.double, requires_grad=False).unsqueeze(0).unsqueeze(0)
            min_dist = loss(input1, input2)
            print(min_dist)
        else:
            loss = RSIL(i_v_shift=3, i_h_shift=3, i_angle=3).train()
            input1 = torch.randn([32, 32], dtype=torch.double, requires_grad=True).unsqueeze(0).unsqueeze(0)
            input2 = torch.randn([32, 32], dtype=torch.double, requires_grad=True).unsqueeze(0).unsqueeze(0)
            min_dist = loss(input1, input2)
            print(min_dist)
            test = gradcheck(loss, [input1, input2])
            print("Are the gradients of whole image rotation and translation with undeformable correct: ", test)
