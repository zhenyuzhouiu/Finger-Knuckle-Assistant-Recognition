# =========================================================
# @ Protocol File:
#
# @ Target dataset: One-session Finger Knuckle Dataset
# @ Parameter Settings:
#       save_mmat:  whether save matching matrix or not,
#                   could be helpful for plot CMC
#
# @ Notes: G-Score: subjects * samples of each subject
# The G-Score will select the minimal scores of each probe
# when compare the rest samples of the same subjects
#          I-Score: subjects * (subjects-1) * samples
# The I-Score will select the minimal scores of each probe
# when compare samples of different subjects
# =========================================================

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from PIL import Image
import numpy as np
import math
import cv2
import torch
import argparse
from torch.autograd import Variable
import models.loss_function, models.net_model
import models.EfficientNetV2
from protocol_util import *
from torchvision import transforms
from inspect import getsourcefile
import os.path as path
from os.path import join
from models.EfficientNetV2 import efficientnetv2_s, ConvBNAct
from collections import OrderedDict
from functools import partial

current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

transform = transforms.Compose([transforms.ToTensor()])

def calc_feats(path):
    """
    1.Read a image from given a path
    2.Normalize image from 0-255 to 0-1
    3.Get the feature map from the model with inference()
    """
    size = args.default_size
    w, h = size[0], size[1]
    container = np.zeros((1, 3, h, w))
    im = np.array(
        Image.open(path).convert("RGB"),
        dtype=np.float32
    )
    container[0, 0, :, :] = im
    container /= 255.
    container = torch.from_numpy(container.astype(np.float32))
    container = container.cuda()
    container = Variable(container, requires_grad=False)
    fv = inference(container)
    return fv.cpu().data.numpy()


def calc_feats_more(*paths, size=(208, 184)):
    """
    1.Read a batch of images from the given paths
    2.Normalize image from 0-255 to 0-1
    3.Get a batch of feature from the model inference()
    """
    size = args.default_size
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
    fv = inference(container)
    # traced_script_module = torch.jit.trace(inference, container)
    # traced_script_module.save("traced_450.pt")

    return fv.cpu().data.numpy()


def genuine_imposter(test_path):
    subs = subfolders(test_path, preserve_prefix=True)
    nsubs = len(subs)
    feats_all = []
    subimnames = []
    for i, usr in enumerate(subs):
        subims = subimages(usr, preserve_prefix=True)
        subimnames += subims
        nims = len(subims)
        feats_all.append(calc_feats_more(*subims))
    feats_all = torch.from_numpy(np.concatenate(feats_all, 0)).cuda()

    # nsubs-> how many subjects on the test_path
    # nims-> how many images on each of subjects' path
    # for example, for hd(1-4), matching_matrix.shape = (714x4, 714x4)
    matching_matrix = np.ones((nsubs * nims, nsubs * nims)) * 1000000
    for i in range(1, feats_all.size(0)):
        loss = _loss(feats_all[:-i, :, :, :], feats_all[i:, :, :, :])
        matching_matrix[:-i, i] = loss
        print("[*] Pre-processing matching dict for {} / {} \r".format(i, feats_all.size(0)))
        # sys.stdout.write("[*] Pre-processing matching dict for {} / {} \r".format(i, feats_all.size(0)))
        # sys.stdout.flush()

    matt = np.ones_like(matching_matrix) * 1e5
    matt[0, :] = matching_matrix[0, :]
    for i in range(1, feats_all.size(0)):
        matt[i, i:] = matching_matrix[i, :-i]
        for j in range(i):
            matt[i, j] = matching_matrix[j, i - j]
    print("\n [*] Done")

    g_scores = []
    i_scores = []
    for i in range(nsubs * nims):
        start_idx = int(math.floor(i / nims))
        start_remainder = int(i % nims)

        argmin_idx = np.argmin(matt[i, start_idx * nims: start_idx * nims + nims])
        g_scores.append(float(matt[i, start_idx * nims + argmin_idx]))
        select = list(range(nsubs * nims))
        # remove genuine matching score
        for j in range(nims):
            select.remove(start_idx * nims + j)
        # remove imposter matching scores of same index sample on other subjects
        for j in range(nsubs):
            if j == start_idx:
                continue
            select.remove(j * nims + start_remainder)
        i_scores += list(np.min(np.reshape(matt[i, select], (-1, nims - 1)), axis=1))
        print("[*] Processing genuine imposter for {} / {} \r".format(i, nsubs * nims))
        # sys.stdout.write("[*] Processing genuine imposter for {} / {} \r".format(i, nsubs * nims))
        # sys.stdout.flush()
    print("\n [*] Done")
    return np.array(g_scores), np.array(i_scores), matt


parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str,
                    default="/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/mask-seg/01/",
                    dest="test_path")
parser.add_argument("--out_path", type=str,
                    default="/media/zhenyuzhou/Data/Project/Finger-Knuckle-2018/Finger-Knuckle-Assistant-Recognition/checkpoint/Joint-Finger-RFNet/Mask-Left-Middle_RFNet-wholeimagerotationandtranslation-lr0.001-subs8-angle0-a20-hs0_vs0_2022-11-16-00-31/output/01-protocol.npy",
                    dest="out_path")
parser.add_argument("--model_path", type=str,
                    default="/media/zhenyuzhou/Data/Project/Finger-Knuckle-2018/Finger-Knuckle-Assistant-Recognition/checkpoint/Joint-Finger-RFNet/Mask-Left-Middle_RFNet-wholeimagerotationandtranslation-lr0.001-subs8-angle0-a20-hs0_vs0_2022-11-16-00-31/ckpt_epoch_3000.pth",
                    dest="model_path")
parser.add_argument("--default_size", type=int, dest="default_size", default=(208, 184))
parser.add_argument("--shift_size", type=int, dest="shift_size", default=0)
parser.add_argument('--block_size', type=int, dest="block_size", default=8)
parser.add_argument("--rotate_angle", type=int, dest="rotate_angle", default=0)
parser.add_argument("--top_k", type=int, dest="top_k", default=16)
parser.add_argument("--save_mmat", type=bool, dest="save_mmat", default=True)
parser.add_argument('--model', type=str, dest='model', default="RFNet")

model_dict = {
    "RFNet": models.net_model.ResidualFeatureNet().cuda(),
    "DeConvRFNet": models.net_model.DeConvRFNet().cuda(),
    "EfficientNetV2-S": models.EfficientNetV2.efficientnetv2_s().cuda(),
    "RFNWithSTNet": models.net_model.RFNWithSTNet().cuda(),
    "ConvNet": models.net_model.ConvNet().cuda(),
}

args = parser.parse_args()
inference = model_dict[args.model].cuda()
if args.model == "EfficientNetV2-S":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_trained = "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/" \
                  "Finger-Knuckle-Assistant-Recognition/checkpoint/EfficientNetV2-S/pre_efficientnetv2-s.pth"
    weights_dict = torch.load(pre_trained, map_location=device)
    load_weights_dict = {k: v for k, v in weights_dict.items()
                         if inference.state_dict()[k].numel() == v.numel()}
    print(inference.load_state_dict(load_weights_dict, strict=False))
    norm_layer = partial(torch.nn.BatchNorm2d, eps=1e-3, momentum=0.1)
    fk_head = OrderedDict()
    # head_input_c of efficientnet_s: 512
    fk_head.update({"conv1": ConvBNAct(256,
                                       64,
                                       kernel_size=3,
                                       norm_layer=norm_layer)})  # ?????????????????????SiLU

    fk_head.update({"conv2": ConvBNAct(64,
                                       1,
                                       kernel_size=3,
                                       norm_layer=norm_layer)})

    fk_head = torch.nn.Sequential(fk_head)
    inference.head = fk_head
    inference = inference.cuda().eval()

inference.load_state_dict(torch.load(args.model_path))
# inference = torch.jit.load("knuckle-script-polyu.pt")
# Loss = models.loss_function.ShiftedLoss(args.shift_size, args.shift_size)
Loss = models.loss_function.WholeImageRotationAndTranslation(args.shift_size, args.shift_size, args.rotate_angle)
# Loss = models.loss_function.ImageBlockRotationAndTranslation(i_block_size=args.block_size, i_v_shift=args.shift_size,
#                                                              i_h_shift=args.shift_size, i_angle=args.rotate_angle,
#                                                              i_topk=args.top_k)
Loss.cuda()
Loss.eval()


def _loss(feats1, feats2):
    loss = Loss(feats1, feats2)
    if isinstance(loss, torch.autograd.Variable):
        loss = loss.data
    return loss.cpu().numpy()


inference = inference.cuda()
inference.eval()

gscores, iscores, mmat = genuine_imposter(args.test_path)
if args.save_mmat:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores, "mmat": mmat})
else:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores})
