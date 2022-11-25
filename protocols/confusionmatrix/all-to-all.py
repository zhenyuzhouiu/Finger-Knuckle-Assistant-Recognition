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
from models.pytorch_mssim import SSIM
import os.path as path
from os.path import join

current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
transform = transforms.Compose([transforms.ToTensor()])


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
    mask = np.zeros((len(paths), 1, int(h/4), int(w/4)))
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
        ma = np.ones([1, int(size[1]/4), int(size[0]/4)])
        mask[i, :, :, :] = ma
    container /= 255.
    container = torch.from_numpy(container.astype(np.float32))
    container = container.cuda()
    container = Variable(container, requires_grad=False)
    mask = torch.from_numpy(mask.astype(np.float32))
    mask = mask.cuda()
    mask = Variable(mask, requires_grad=False)
    fv, mask = inference(container, mask)
    # traced_script_module = torch.jit.trace(inference, container)
    # traced_script_module.save("traced_450.pt")

    return fv.cpu().data.numpy(), mask.cpu().data.numpy()

def genuine_imposter(test_path):
    subs = subfolders(test_path, preserve_prefix=True)
    subs.sort()
    feats_all = []
    feats_length = []
    nfeats = 0
    for i, usr in enumerate(subs):
        subims = subimages(usr, preserve_prefix=True)
        subims.sort()
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


def genuine_imposter_upright(test_path):
    subs = subfolders(test_path, preserve_prefix=True)
    subs.sort()
    feats_all = []
    mask_all = []
    feats_length = []
    nfeats = 0
    for i, usr in enumerate(subs):
        subims = subimages(usr, preserve_prefix=True)
        subims.sort()
        nfeats += len(subims)
        feats_length.append(len(subims))
        fm, ma = calc_feats_more(*subims)
        feats_all.append(fm)
        mask_all.append(ma)
    feats_length = np.array(feats_length)
    acc_len = np.cumsum(feats_length)
    feats_start = acc_len - feats_length

    feats_all = torch.from_numpy(np.concatenate(feats_all, 0)).cuda()
    mask_all = torch.from_numpy(np.concatenate(mask_all, 0)).cuda()
    matching_matrix = np.ones((nfeats, nfeats)) * 1e5
    for i in range(1, feats_all.size(0)):
        loss = _loss(feats_all[:-i, :, :, :], mask_all[:-i, :, :, :], feats_all[i:, :, :, :], mask_all[i:, :, :, :])
        matching_matrix[:-i, i] = loss
        print("[*] Pre-processing matching dict for {} / {} \r".format(i, feats_all.size(0)))

    matt = np.ones_like(matching_matrix) * 1e5
    matt[0, :] = matching_matrix[0, :]
    for i in range(1, feats_all.size(0)):
        matt[i, i:] = matching_matrix[i, :-i]
        # for j in range(i):
        #     matt[i, j] = matching_matrix[j, i - j]
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
                    default="/media/zhenyuzhou/Data/Project/Finger-Knuckle-2018/Finger-Knuckle-Assistant-Recognition/checkpoint/Joint-Finger-RFNet/ssim/MaskLM_RFNet_triplet_ssim-lr0.001-r0-a1-2a20-hs0_vs0_11-24-21-04-13/output/04-protocol.npy",
                    dest="out_path")
parser.add_argument("--model_path", type=str,
                    default="/media/zhenyuzhou/Data/Project/Finger-Knuckle-2018/Finger-Knuckle-Assistant-Recognition/checkpoint/Joint-Finger-RFNet/ssim/MaskLM_RFNet_triplet_ssim-lr0.001-r0-a1-2a20-hs0_vs0_11-24-21-04-13/ckpt_epoch_3000.pth",
                    dest="model_path")
parser.add_argument("--default_size", type=int, dest="default_size", default=(128, 128))
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
    "STNWithRFNet": models.net_model.STNWithRFNet().cuda(),
    "ResidualSTNet": models.net_model.ResidualSTNet().cuda(),
    "RFNet64": models.net_model.RFNet64().cuda()
}

args = parser.parse_args()
inference = model_dict[args.model].cuda()

inference.load_state_dict(torch.load(args.model_path))
# Loss = models.loss_function.WholeRotationShiftedLoss(args.shift_size, args.shift_size, args.angle)
# Loss = models.loss_function.MaskRSIL(args.shift_size, args.shift_size, args.rotate_angle)
Loss = SSIM(data_range=1., size_average=False, channel=1)
Loss.cuda()
Loss.eval()


def _loss(feats1, mask1, feats2, mask2):
    loss = Loss(feats1, mask1, feats2, mask2)
    if isinstance(loss, torch.autograd.Variable):
        loss = loss.data
    return loss.cpu().numpy()


inference = inference.cuda()
inference.eval()

gscores, iscores, _, mmat = genuine_imposter_upright(args.test_path)

if args.save_mmat:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores, "mmat": mmat})
else:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores})
