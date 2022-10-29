# =========================================================
# @ Protocol File: Two sessions (Probe / Gallery)
#
# @ Target dataset:
# @ Parameter Settings:
#       save_mmat:  whether save matching matrix or not,
#                   could be helpful for plot CMC
#
# @ Notes:  also could be used on PolyU 2D if format the
#           dataset like "session1" and "session2" under
#           PolyU 2D data folder
# =========================================================

from __future__ import division
import os
import sys
import time
from PIL import Image
import numpy as np
import torch
import math
import argparse
from torch.autograd import Variable

from inspect import getsourcefile

current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import models.loss_function, models.net_model
from protocol_util import *
from protocol_util import *
import models.EfficientNetV2

from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()])


def calc_feats(path):
    size = args.default_size
    w, h = size[0], size[1]
    container = np.zeros((1, 3, h, w))
    im = np.array(
        Image.open(path).convert("RGB").resize(size=size),
        dtype=np.float32
    )
    container[0, 0, :, :] = im
    container /= 255.
    container = torch.from_numpy(container.astype(np.float32))
    container = container.cuda()
    container = Variable(container, requires_grad=False)
    fv = inference(container)
    return fv.cpu().data.numpy()


def calc_feats_more(*paths):
    size = args.default_size
    w, h =size[0], size[1]
    container = np.zeros((len(paths), 3, h, w))
    for i, path in enumerate(paths):
        im = np.array(
            Image.open(path).convert("RGB").resize(size=size),
            dtype=np.float32
        )
        im = np.transpose(im, (2, 0, 1))
        container[i, :, :, :] = im
    container /= 255.
    container = torch.from_numpy(container.astype(np.float32))
    container = container.cuda()
    container = Variable(container, requires_grad=False)
    fv = inference(container)
    return fv.cpu().data.numpy()


def genuine_imposter(args_session1_path, args_session2_path):
    session1_path = args_session1_path
    session2_path = args_session2_path

    subs_session1 = subfolders(session1_path, preserve_prefix=True)
    subs_session1 = sorted(subs_session1)
    subs_session2 = subfolders(session2_path, preserve_prefix=True)
    subs_session2 = sorted(subs_session2)
    nsubs1 = len(subs_session1)
    nsubs2 = len(subs_session2)
    assert (nsubs1 == nsubs2 and nsubs1 != 0)

    nsubs = nsubs1
    nims = -1
    feats_probe = []
    feats_gallery = []

    for gallery, probe in zip(subs_session1, subs_session2):
        assert (os.path.basename(gallery) == os.path.basename(probe))
        im_gallery = subimages(gallery, preserve_prefix=True)
        im_probe = subimages(probe, preserve_prefix=True)

        nim_gallery = len(im_gallery)
        nim_probe = len(im_probe)
        if nims == -1:
            nims = nim_gallery
            assert (nims == nim_probe)  # Check if image numbers in probe equals number in gallery
        else:
            assert (nims == nim_gallery and nims == nim_probe)  # Check for each folder

        probe_fv = calc_feats_more(*im_probe)
        gallery_fv = calc_feats_more(*im_gallery)

        feats_probe.append(probe_fv)
        feats_gallery.append(gallery_fv)

    feats_probe = torch.from_numpy(np.concatenate(feats_probe, 0)).cuda()
    feats_gallery = np.concatenate(feats_gallery, 0)
    feats_gallery2 = np.concatenate((feats_gallery, feats_gallery), 0)
    feats_gallery = torch.from_numpy(feats_gallery2).cuda()

    nl = nsubs * nims
    matching_matrix = np.ones((nl, nl)) * 1000000
    for i in range(nl):
        loss = _loss(feats_probe, feats_gallery[i: i + nl, :, :, :])
        matching_matrix[:, i] = loss
        print("[*] Pre-processing matching dict for {} / {} \r".format(i, nl))
        # sys.stdout.write("[*] Pre-processing matching dict for {} / {} \r".format(i, nl))
        # sys.stdout.flush()

    for i in range(1, nl):
        tmp = matching_matrix[i, -i:].copy()
        matching_matrix[i, i:] = matching_matrix[i, :-i]
        matching_matrix[i, :i] = tmp
    print("\n [*] Done")

    g_scores = []
    i_scores = []
    for i in range(nl):
        start_idx = int(math.floor(i / nims))
        start_remainder = int(i % nims)
        g_scores.append(float(np.min(matching_matrix[i, start_idx * nims: start_idx * nims + nims])))
        select = list(range(nl))
        for j in range(nims):
            select.remove(start_idx * nims + j)
        i_scores += list(np.min(np.reshape(matching_matrix[i, select], (-1, nims)), axis=1))
        print("[*] Processing genuine imposter for {} / {} \r".format(i, nsubs * nims))
        # sys.stdout.write("[*] Processing genuine imposter for {} / {} \r".format(i, nsubs * nims))
        # sys.stdout.flush()
    print("\n [*] Done")
    return np.array(g_scores), np.array(i_scores), matching_matrix


parser = argparse.ArgumentParser()
parser.add_argument("--session1", type=str,
                    default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/Finger-Knuckle-Assistant-Recognition/dataset/PolyUKnuckleV3/yolov5/Session_1/1-104/",
                    dest="session1")
parser.add_argument("--session2", type=str,
                    default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/Finger-Knuckle-Assistant-Recognition/dataset/PolyUKnuckleV3/yolov5/Session_2/",
                    dest="session2")
parser.add_argument("--out_path", type=str,
                    default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/Finger-Knuckle-Assistant-Recognition/checkpoint/RFNet-TL/fkv3(yolov5)-105-221_RFNet-wholeimagerotationandtranslation-lr0.001-subs8-angle0-a20-hs0_vs0_2022-07-18-10-16/output/yolov5-protocol.npy",
                    dest="out_path")
parser.add_argument("--model_path", type=str,
                    default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/Finger-Knuckle-Assistant-Recognition/checkpoint/RFNet-TL/fkv3(yolov5)-105-221_RFNet-wholeimagerotationandtranslation-lr0.001-subs8-angle0-a20-hs0_vs0_2022-07-18-10-16/ckpt_epoch_6000.pth",
                    dest="model_path")
parser.add_argument('--default_size', type=int, dest='default_size', default=(128, 128))
parser.add_argument('--horizontal_size', type=int, dest='horizontal_size', default=0)
parser.add_argument('--vertical_size', type=int, dest='vertical_size', default=0)
parser.add_argument('--block_size', type=int, dest="block_size", default=8)
parser.add_argument("--rotate_angle", type=int, dest="rotate_angle", default=0)
parser.add_argument("--top_k", type=int, dest="top_k", default=16)
parser.add_argument("--save_mmat", type=bool, dest="save_mmat", default=True)
parser.add_argument('--model', type=str, dest='model', default="RFNet")

model_dict = {
    "RFNet": models.net_model.ResidualFeatureNet().cuda(),
    "DeConvRFNet": models.net_model.DeConvRFNet().cuda(),
    "EfficientNet": models.EfficientNetV2.fk_efficientnetv2_s().cuda(),
    "RFNWithSTNet": models.net_model.RFNWithSTNet().cuda(),
    "ConvNet": models.net_model.ConvNet().cuda(),
}

args = parser.parse_args()
inference = model_dict[args.model].cuda()
inference.load_state_dict(torch.load(args.model_path))
# Loss = models.loss_function.ShiftedLoss(args.shift_size, args.shift_size)
Loss = models.loss_function.WholeImageRotationAndTranslation(args.vertical_size, args.horizontal_size, args.rotate_angle)
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

gscores, iscores, mmat = genuine_imposter(args.session1, args.session2)
if args.save_mmat:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores, "mmat": mmat})
else:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores})
