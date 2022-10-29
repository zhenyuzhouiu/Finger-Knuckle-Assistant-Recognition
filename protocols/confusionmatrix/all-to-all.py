# =========================================================
# @ Protocol File: All-to-All protocols
#
# @ Target dataset:
# @ Parameter Settings:
#       save_mmat:  whether save matching matrix or not,
#                   could be helpful for plot CMC
# @ Note: G-Scores: Subjects * (Samples * (Samples-1)) / 2
#                   or Subjects * (Samples * Samples)
#         I-Scores: Subjects * (Subject-1) * (Samples * Samples)
# =========================================================
import os
import sys
from PIL import Image
import numpy as np
import torch
import argparse
from torch.autograd import Variable
import models.EfficientNetV2
import models.loss_function, models.net_model
from protocol_util import *
from torchvision import transforms
from inspect import getsourcefile
import os.path as path
from os.path import join

current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
transform = transforms.Compose([transforms.ToTensor()])


def calc_feats_more(*paths):
    container = np.zeros((len(paths), 3, args.default_size, args.default_size))
    for i, path in enumerate(paths):
        im = np.array(
            Image.open(path).convert("RGB").resize((args.default_size, args.default_size)),
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


def genuine_imposter(test_path):
    subs = subfolders(test_path, preserve_prefix=True)
    feats_all = []
    feats_length = []
    nfeats = 0
    for i, usr in enumerate(subs):
        subims = subimages(usr, preserve_prefix=True)
        nfeats += len(subims)
        feats_length.append(len(subims))
        feats_all.append(calc_feats_more(*subims))
    feats_length = np.array(feats_length)
    acc_len = np.cumsum(feats_length)
    feats_start = acc_len - feats_length

    feats_all = torch.from_numpy(np.concatenate(feats_all, 0)).cuda()
    matching_matrix = np.ones((nfeats, nfeats)) * 1e5
    for i in range(1, feats_all.size(0)):
        loss = _loss(feats_all[:-i, :, :, :], feats_all[i:, :, :, :])
        matching_matrix[:-i, i] = loss
        print("[*] Pre-processing matching dict for {} / {} \r".format(i, feats_all.size(0)))
        # sys.stdout.write("[*] Pre-processing matching dict for {} / {} \r".format(i, feats_all.size(0)))
        # sys.stdout.flush()

    mmat = np.ones_like(matching_matrix) * 1e5
    mmat[0, :] = matching_matrix[0, :]
    for i in range(1, feats_all.size(0)):
        mmat[i, i:] = matching_matrix[i, :-i]
        for j in range(i):
            mmat[i, j] = matching_matrix[j, i - j]
    print("\n [*] Done")

    g_scores = []
    i_scores = []
    for i in range(nfeats):
        subj_idx = np.argmax(acc_len > i)
        g_select = [feats_start[subj_idx] + k for k in range(feats_length[subj_idx])]
        g_select.remove(i)
        i_select = list(range(nfeats))
        for k in range(feats_length[subj_idx]):
            i_select.remove(feats_start[subj_idx] + k)
        g_scores += list(mmat[i, g_select])
        i_scores += list(mmat[i, i_select])

    print("\n [*] Done")
    return np.array(g_scores), np.array(i_scores), feats_length, mmat


parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str,
                    default="/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/right-yolov5x-csl/right-little-resize/",
                    dest="test_path")
parser.add_argument("--out_path", type=str,
                    default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/Finger-Knuckle-Assistant-Recognition/checkpoint/Joint-Finger-RFNet/Joint-Left-Middle_RFNet-wholeimagerotationandtranslation-lr0.001-subs8-angle4-a20-hs4_vs4_2022-10-20-17-25/output/right-little.npy",
                    dest="out_path")
parser.add_argument("--model_path", type=str,
                    default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/Finger-Knuckle-Assistant-Recognition/checkpoint/Joint-Finger-RFNet/Joint-Left-Middle_RFNet-wholeimagerotationandtranslation-lr0.001-subs8-angle4-a20-hs4_vs4_2022-10-20-17-25/ckpt_epoch_3820.pth",
                    dest="model_path")
parser.add_argument("--default_size", type=int, dest="default_size", default=128)
parser.add_argument("--shift_size", type=int, dest="shift_size", default=4)
parser.add_argument('--block_size', type=int, dest="block_size", default=8)
parser.add_argument("--rotate_angle", type=int, dest="rotate_angle", default=4)
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

inference.load_state_dict(torch.load(args.model_path))
# Loss = models.loss_function.WholeRotationShiftedLoss(args.shift_size, args.shift_size, args.angle)
Loss = models.loss_function.WholeImageRotationAndTranslation(args.shift_size, args.shift_size, args.rotate_angle)
# Loss = models.loss_function.ImageBlockRotationAndTranslation(args.block_size, args.shift_size, args.shift_size,
#                                                              args.rotate_angle, args.top_k)
Loss.cuda()
Loss.eval()


def _loss(feats1, feats2):
    loss = Loss(feats1, feats2)
    if isinstance(loss, torch.autograd.Variable):
        loss = loss.data
    return loss.cpu().numpy()


inference = inference.cuda()
inference.eval()

gscores, iscores, _, mmat = genuine_imposter(args.test_path)

if args.save_mmat:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores, "mmat": mmat})
else:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores})
