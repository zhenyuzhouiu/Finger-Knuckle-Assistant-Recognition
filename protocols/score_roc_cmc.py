import os
from PIL import Image
import numpy as np
import scipy.io as io
import torch
import cv2
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.net_model import ResidualFeatureNet, RFNet64, SERFNet64, \
    STNRFNet64, STNResRFNet64, STNResRFNet64v2, STNResRFNet64v3, DeformRFNet64, \
    DilateRFNet64, RFNet64Relu, STNResRFNet64v2Relu, STNResRFNet32v216, STNResRFNet32v316, STNResRFNet3v316, STNResRFNet3v216
from torch.autograd import Variable
from protocols.plot.plotroc_basic import *
from protocols.confusionmatrix.protocol_util import *
from models.pytorch_mssim import SSIM, RSSSIM
from loss.stn_ssim import STSSIM
from models.loss_function import ShiftedLoss
import matplotlib.pyplot as plt
import matplotlib
import tqdm

model_dict = {
    "RFNet": ResidualFeatureNet(),
    "RFNet64": RFNet64(),
    "SERFNet64": SERFNet64(),
    "STNRFNet64": STNRFNet64(),
    "STNResRFNet64": STNResRFNet64(),
    "STNResRFNet64v2": STNResRFNet64v2(),
    "STNResRFNet64v3": STNResRFNet64v3(),
    "DeformRFNet64": DeformRFNet64(),
    "DilateRFNet64": DilateRFNet64(),
    "RFNet64Relu": RFNet64Relu(),
    "STNResRFNet64v2Relu": STNResRFNet64v2Relu(),
    "STNResRFNet32v216": STNResRFNet32v216(),
    "STNResRFNet32v316": STNResRFNet32v316(),
    "STNResRFNet3v316": STNResRFNet3v316(),
    "STNResRFNet3v216": STNResRFNet3v216()
}


def _loss(feats1, feats2, loss_model):
    # loss = Loss(feats1, mask1, feats2, mask2)
    loss = loss_model(feats1, feats2)
    if isinstance(loss, torch.autograd.Variable):
        loss = loss.data
    return loss.cpu().numpy()


def calc_feats_more(*paths, size=(128, 128), options='RGB', model, gpu_num):
    """
    1.Read a batch of images from the given paths
    2.Normalize image from 0-255 to 0-1
    3.Get a batch of feature from the model inference()
    """
    if options == 'RGB':
        w, h = size[0], size[1]
        ratio = size[1] / size[0]
        container = np.zeros((len(paths), 3, h, w))
        # mask = np.zeros((len(paths), 1, int(h / 4), int(w / 4)))
        for i, path in enumerate(paths):
            image = np.array(
                Image.open(path).convert(options),
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
        container = container.cuda(gpu_num)
        container = Variable(container, requires_grad=False)
        fv = model(container)
    else:
        w, h = size[0], size[1]
        ratio = size[1] / size[0]
        container = np.zeros((len(paths), 1, h, w))
        # mask = np.zeros((len(paths), 1, int(h / 4), int(w / 4)))
        for i, path in enumerate(paths):
            image = np.array(
                Image.open(path).convert(options),
                dtype=np.float32
            )
            image = image[8:-8, :]
            h, w = image.shape
            dest_w = h / ratio
            dest_h = w * ratio
            if dest_w > w:
                crop_h = int((h - dest_h) / 2)
                if crop_h == 0:
                    crop_h = 1
                crop_image = image[crop_h - 1:crop_h + int(dest_h), :]
            elif dest_h > h:
                crop_w = int((w - dest_w) / 2)
                if crop_w == 0:
                    crop_w = 1
                crop_image = image[:, crop_w - 1:crop_w + int(dest_w)]
            else:
                crop_image = image
            resize_image = cv2.resize(crop_image, dsize=size)
            container[i, 0, :, :] = resize_image
        container /= 255.
        container = torch.from_numpy(container.astype(np.float32))
        container = container.cuda(gpu_num)
        container = Variable(container, requires_grad=False)
        fv = model(container)
    return fv.cpu().data.numpy()


def genuine_imposter_upright(test_path, image_size, options, inference, loss_model, gpu_num):
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
        fm = calc_feats_more(*subims, size=image_size, options=options, model=inference, gpu_num=gpu_num)
        feats_all.append(fm)
    feats_length = np.array(feats_length)
    acc_len = np.cumsum(feats_length)
    feats_start = acc_len - feats_length

    feats_all = torch.from_numpy(np.concatenate(feats_all, 0)).cuda(gpu_num)
    matching_matrix = np.ones((nfeats, nfeats)) * 1e5

    for i in tqdm.tqdm(range(1, feats_all.size(0))):
        x = feats_all[:-i, :, :, :]
        y = feats_all[i:, :, :, :]
        bs, ch, he, wi = x.shape
        loss = np.ones(bs, ) * 1e5
        chuncks = 6000
        if bs > chuncks:
            num_chuncks = bs // chuncks
            num_reminder = bs % chuncks
            for nc in range(num_chuncks):
                x_nc = x[0 + nc * chuncks:chuncks + nc * chuncks, :, :, :]
                y_nc = y[0 + nc * chuncks:chuncks + nc * chuncks, :, :, :]
                loss[0 + nc * chuncks:chuncks + nc * chuncks] = _loss(x_nc, y_nc, loss_model=loss_model)
            if num_reminder > 0:
                x_nc = x[chuncks + nc * chuncks:, :, :, :]
                y_nc = y[chuncks + nc * chuncks:, :, :, :]
                if x_nc.ndim == 3:
                    x_nc = x_nc.unsqueeze(0)
                    y_nc = y_nc.unsqueeze(0)
                loss[chuncks + nc * chuncks:] = _loss(x_nc, y_nc, loss_model=loss_model)
        else:
            loss = _loss(feats_all[:-i, :, :, :], feats_all[i:, :, :, :], loss_model=loss_model)
        matching_matrix[:-i, i] = loss

    matt = np.ones_like(matching_matrix) * 1e5
    matt[0, :] = matching_matrix[0, :]
    for i in range(1, feats_all.size(0)):
        matt[i, i:] = matching_matrix[i, :-i]
        # for j in range(i):
        #     matt[i, j] = matching_matrix[j, i - j]

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
    g_scores = np.array(g_scores)
    i_scores = np.array(i_scores)
    g_min = np.min(g_scores)
    g_max = np.max(g_scores)
    i_min = np.min(i_scores)
    i_max = np.max(i_scores)
    min = np.min(np.array([g_min, i_min]))
    max = np.max(np.array([g_max, i_max]))
    g_scores = (g_scores - min) / (max - min)
    i_scores = (i_scores - min) / (max - min)

    return np.array(g_scores), np.array(i_scores), matt


def draw_roc(src_mat, color, label, dst):
    for i in range(len(src_mat)):
        data = io.loadmat(src_mat[i])
        g_scores = np.array(data['g_scores'])
        g_scores = np.squeeze(g_scores)
        i_scores = np.array(data['i_scores'])
        i_scores = np.squeeze(i_scores)

        print('[*] Source file: {}'.format(src_mat[i]))
        print('[*] Target output file: {}'.format(dst))
        print("[*] #Genuine: {}\n[*] #Imposter: {}".format(len(g_scores), len(i_scores)))

        x, y = calc_coordinates(g_scores, i_scores)
        print("[*] EER: {}".format(calc_eer(x, y)))
        EER = "%.3f%%" % (calc_eer(x, y) * 100)

        # xmin, xmax = plt.xlim()
        # ymin, ymax = plt.ylim()

        lines = plt.plot(x, y, label='ROC')
        plt.setp(lines, 'color', color[i], 'linewidth', 3, 'label', label[i] + "; EER: " + str(EER))

        matplotlib.rc('xtick', labelsize=10)
        matplotlib.rc('ytick', labelsize=10)

        plt.grid(True)
        plt.xlabel(r'False Accept Rate', fontsize=18)
        plt.ylabel(r'Genuine Accept Rate', fontsize=18)
        legend = plt.legend(loc='lower right', shadow=False, prop={'size': 16})
        plt.xlim(xmin=min(x))
        plt.xlim(xmax=0)
        plt.ylim(ymax=1)
        plt.ylim(ymin=0.4)

        ax = plt.gca()
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2.0)

        plt.xticks(np.array([-4, -2, 0]), ['$10^{-4}$', '$10^{-2}$', '$10^{0}$'], fontsize=16)
        plt.yticks(np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), fontsize=16)

    dst = os.path.join(dst, "roc.pdf")
    plt.savefig(dst, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str,
                        default="/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/mask-seg/",
                        dest="test_path")
    parser.add_argument("--out_path", type=str,
                        default="../checkpoint/Joint-Finger-RFNet/MaskLM_STNResRFNet3v216_quadruplet_rsssim_12-29-20-55-14/output/",
                        dest="out_path")
    parser.add_argument("--model_path", type=str,
                        default="../checkpoint/Joint-Finger-RFNet/MaskLM_STNResRFNet3v216_quadruplet_rsssim_12-29-20-55-14/ckpt_epoch_1460.pth",
                        dest="model_path")
    parser.add_argument("--loss_path", type=str,
                        default="../checkpoint/Joint-Finger-RFNet/MaskLM_STNResRFNet3v216_quadruplet_rsssim_12-29-20-55-14/loss_epoch_3000.pth",
                        dest="loss_path")
    parser.add_argument('--model', type=str, dest='model', default="STNResRFNet3v216")
    parser.add_argument('--loss', type=str, dest='loss', default="RSSSIM")
    parser.add_argument("--default_size", type=int, dest="default_size", default=(128, 128))
    parser.add_argument("--option", type=str, dest="option", default='RGB')
    parser.add_argument("--v_shift", type=int, dest="v_shift", default=4)
    parser.add_argument("--h_shift", type=int, dest="h_shift", default=4)
    parser.add_argument("--rotate_angle", type=int, dest="rotate_angle", default=4)
    parser.add_argument("--step_size", type=int, dest="step_size", default=1)
    parser.add_argument("--save_mmat", type=bool, dest="save_mmat", default=True)
    parser.add_argument("--gpu_num", type=int, dest="gpu_num", default=0)
    parser.add_argument("--if_draw", type=bool, dest="if_draw", default=True)
    args = parser.parse_args()

    cls_num = ['01', '02', '04', '07']

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    inference = model_dict[args.model].cuda(args.gpu_num)
    inference.load_state_dict(torch.load(args.model_path))
    inference.eval()

    if args.loss == "SSIM":
        Loss = SSIM(data_range=1., size_average=False, win_size=7, channel=3)
    elif args.loss == "RSSSIM":
        Loss = RSSSIM(data_range=1., size_average=False, win_size=7, channel=3, v_shift=args.v_shift,
                      h_shift=args.h_shift, angle=args.rotate_angle, step=args.step_size)
    else:
        if args.loss == "STSSIM":
            Loss = STSSIM(data_range=1., size_average=False, channel=64)
        else:
            Loss = ShiftedLoss(hshift=args.h_shift, vshift=args.v_shift)
    Loss.cuda(args.gpu_num)
    if args.loss == "STSSIM":
        Loss.load_state_dict(torch.load(args.loss_path))
    Loss.eval()

    for c in cls_num:
        test_path = os.path.join(args.test_path, c)
        out_path = os.path.join(args.out_path, c + ".mat")
        gscores, iscores, mmat = genuine_imposter_upright(test_path=test_path, image_size=args.default_size,
                                                          options= args.option, inference=inference, loss_model=Loss, gpu_num= args.gpu_num)

        if args.save_mmat:
            io.savemat(out_path, {"g_scores": gscores, "i_scores": iscores, "mmat": mmat})
        else:
            io.savemat(out_path, {"g_scores": gscores, "i_scores": iscores})

    if args.if_draw:
        src_mat = []
        label = []
        color = ['#000000',
                 "#c0c0c0",
                 '#ff0000',
                 '#008000',
                 '#008080',
                 '#000080',
                 '#00ffff',
                 '#800000',
                 '#800080',
                 '#808000',
                 '#ff00ff',
                 '#ff0000']
        for c in cls_num:
            src_mat.append(os.path.join(args.out_path, c + ".mat"))
            label.append(c)

        draw_roc(src_mat, color, label, dst=args.out_path)
