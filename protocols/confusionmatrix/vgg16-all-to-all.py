# =========================================================
# @ Protocol File: All-to-All protocols
#
# @ Target dataset:
# @ Parameter Settings:
#       save_mmat:  whether save matching matrix or not,
#                   could be helpful for plot CMC
# @ Note: G-Scores: Subjects * (Samples * (Samples-1)) / 2
#                   or Subjects * (Samples * Samples)
#         I-Scores: Subjects * (Subject-1) * (Samples * Samples) / 2
# =========================================================
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from PIL import Image
import numpy as np
import torch
import cv2
import argparse
from torch.autograd import Variable
import models.EfficientNetV2
import models.loss_function, models.net_model
from protocol_util import *
from torchvision import transforms
from inspect import getsourcefile
from models.pytorch_mssim import SSIM, SSIMGNN, RSSSIM
from models.vgg16 import *
import os.path as path
from os.path import join

current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
transform = transforms.Compose([transforms.ToTensor()])


def calc_feats_more(*paths, size=(128, 128)):
    """
    1.Read a batch of images from the given paths
    2.Normalize image from 0-255 to 0-1
    3.Get a batch of feature from the model inference()
    """
    w, h = size[0], size[1]
    ratio = size[1] / size[0]
    container = np.zeros((len(paths), 3, h, w))
    for i, path in enumerate(paths):
        image = np.array(
            Image.open(path).convert('RGB'),
            dtype=np.float32
        )
        image = image[8:-8, :, :]
        h, w, c = image.shape
        dest_w = h / ratio
        dest_h = w * ratio
        if dest_w > w:
            crop_h = int((h - dest_h) / 2)
            if crop_h == 0:
                crop_h = 1
            crop_image = image[crop_h - 1:crop_h + int(dest_h), :, :]
        elif dest_h > h:
            crop_w = int((w - dest_w) / 2)
            if crop_w == 0:
                crop_w = 1
            crop_image = image[:, crop_w - 1:crop_w + int(dest_w), :]
        else:
            crop_image = image
        resize_image = cv2.resize(crop_image, dsize=size)
        # change hxwxc = cxhxw
        im = np.transpose(resize_image, (2, 0, 1))
        container[i, :, :, :] = im
    container /= 255.
    container = torch.from_numpy(container.astype(np.float32))
    container = container.cuda()
    container = Variable(container, requires_grad=False)
    fv8 = inference8(container)
    fv32 = inference32(container)
    return fv8.cpu().data.numpy(), fv32.cpu().data.numpy()


def genuine_imposter_upright(test_path):
    subs = subfolders(test_path, preserve_prefix=True)
    subs.sort()
    feats_8 = []
    feats_32 = []
    feats_length = []
    nfeats = 0
    for i, usr in enumerate(subs):
        subims = subimages(usr, preserve_prefix=True)
        subims.sort()
        nfeats += len(subims)
        feats_length.append(len(subims))
        fm8, fm32 = calc_feats_more(*subims)
        feats_8.append(fm8)
        feats_32.append(fm32)
    feats_length = np.array(feats_length)
    acc_len = np.cumsum(feats_length)
    feats_start = acc_len - feats_length

    feats_8 = torch.from_numpy(np.concatenate(feats_8, 0)).cuda()
    feats_32 = torch.from_numpy(np.concatenate(feats_32, 0)).cuda()
    matching_8 = np.ones((nfeats, nfeats)) * 1e5
    matching_32 = np.ones((nfeats, nfeats)) * 1e5
    for i in range(1, feats_8.size(0)):
        x_8 = feats_8[:-i, :, :, :]
        y_8 = feats_8[i:, :, :, :]
        x_32 = feats_32[:-i, :, :, :]
        y_32 = feats_32[i:, :, :, :]
        bs, ch, he, wi = x_8.shape
        loss_t = np.ones(bs, ) * 1e5
        loss_k = np.ones(bs, ) * 1e5
        chuncks = 6000
        if bs > chuncks:
            num_chuncks = bs // chuncks
            num_reminder = bs % chuncks
            for nc in range(num_chuncks):
                x8_nc = x_8[0 + nc * chuncks:chuncks + nc * chuncks, :, :, :]
                y8_nc = y_8[0 + nc * chuncks:chuncks + nc * chuncks, :, :, :]
                loss_k[0 + nc * chuncks:chuncks + nc * chuncks] = k_loss(x8_nc, y8_nc)
                x32_nc = x_32[0 + nc * chuncks:chuncks + nc * chuncks, :, :, :]
                y32_nc = y_32[0 + nc * chuncks:chuncks + nc * chuncks, :, :, :]
                loss_t[0 + nc * chuncks:chuncks + nc * chuncks] = t_loss(x32_nc, y32_nc)
            if num_reminder > 0:
                x8_nc = x_8[chuncks + nc * chuncks:, :, :, :]
                y8_nc = y_8[chuncks + nc * chuncks:, :, :, :]
                x32_nc = x_32[chuncks + nc * chuncks:, :, :, :]
                y32_nc = y_32[chuncks + nc * chuncks:, :, :, :]
                if x8_nc.ndim == 3:
                    x8_nc = x8_nc.unsqueeze(0)
                    y8_nc = y8_nc.unsqueeze(0)
                    x32_nc = x32_nc.unsqueeze(0)
                    y32_nc = y32_nc.unsqueeze(0)
                loss_k[chuncks + nc * chuncks:] = k_loss(x8_nc, y8_nc)
                loss_t[chuncks + nc * chuncks:] = t_loss(x32_nc, y32_nc)
        else:
            loss_t = t_loss(feats_32[:-i, :, :, :], feats_32[i:, :, :, :])
            loss_k = k_loss(feats_32[:-i, :, :, :], feats_32[i:, :, :, :])
        matching_8[:-i, i] = loss_k
        matching_32[:-i, i] = loss_t
        print("[*] Pre-processing matching dict for {} / {} \r".format(i, feats_8.size(0)))

    matt8 = np.ones_like(matching_8) * 1e5
    matt8[0, :] = matching_8[0, :]
    for i in range(1, feats_8.size(0)):
        matt8[i, i:] = matching_8[i, :-i]

    matt32 = np.ones_like(matching_32) * 1e5
    matt32[0, :] = matching_32[0, :]
    for i in range(1, feats_32.size(0)):
        matt32[i, i:] = matching_32[i, :-i]
    print("\n [*] Done")

    g_scores = []
    i_scores = []
    # nfeats: number of features
    for i in range(nfeats):
        # in case of multiple occurrences of the maximum values,
        # the indices corresponding to the first occurrence are returned.
        subj_idx = np.argmax(acc_len > i)
        g_select = [feats_start[subj_idx] + k for k in range(feats_length[subj_idx])]
        for i_i in range(i + 1):
            if i_i in g_select:
                g_select.remove(i_i)
        i_select = list(range(nfeats))
        # remove g_select
        for subj_i in range(subj_idx + 1):
            for k in range(feats_length[subj_i]):
                i_select.remove(feats_start[subj_i] + k)
        if len(g_select) != 0:
            g_scores += list(matt[i, g_select])
        if len(i_select) != 0:
            i_scores += list(matt[i, i_select])

    print("\n [*] Done")
    return np.array(g_scores), np.array(i_scores), feats_length, matt


parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str,
                    default="/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/mask-seg/04/",
                    dest="test_path")
parser.add_argument("--out_path", type=str,
                    default="/media/zhenyuzhou/Data/Project/Finger-Knuckle-2018/Finger-Knuckle-Assistant-Recognition/checkpoint/vgg16/04-protocol.npy",
                    dest="out_path")

args = parser.parse_args()

inference32 = FeatureExtraction(last_layer='relu3_3')
inference32.eval()
inference8 = FeatureExtraction(last_layer='relu5_3')
inference8.eval()

texture_l = SSIM(data_range=1., size_average=False, channel=256)
texture_l.cuda()
texture_l.eval()
keypoint_l = FeatureCorrelation()
keypoint_l.cuda()
keypoint_l.eval()


def t_loss(feats1, feats2):
    # loss = Loss(feats1, mask1, feats2, mask2)
    loss = texture_l(feats1, feats2)
    if isinstance(loss, torch.autograd.Variable):
        loss = loss.data
    return loss.cpu().numpy()


def k_loss(feats1, feats2):
    # correlation_tensor.shape:-> [b, 64, 8, 8]
    correlation_tensor = keypoint_l(feats1, feats2)
    b, c, h, w = correlation_tensor.shape
    channel_max, _ = torch.max(correlation_tensor, dim=1)
    batch_sum = 1- torch.sigmoid(torch.sum(channel_max.view(b, -1), dim=1))
    if isinstance(batch_sum, torch.autograd.Variable):
        batch_sum = batch_sum.data
    return batch_sum.cpu().numpy()


gscores, iscores, _, mmat = genuine_imposter_upright(args.test_path)
save_mmat = True
if save_mmat:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores, "mmat": mmat})
else:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores})
