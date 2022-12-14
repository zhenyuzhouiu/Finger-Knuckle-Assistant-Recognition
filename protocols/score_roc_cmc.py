import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from PIL import Image
import numpy as np
import scipy.io as io
import torch
import cv2
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.net_model import ResidualFeatureNet, STResNet_R, STResNet_S, \
    STResNetRelu_R, STResNetRelu_S, STResNet16_R, STResNet16_S, STNResRFNet3v216, ResNet
from torch.autograd import Variable
from protocols.plot.plotroc_basic import *
from protocols.confusionmatrix.protocol_util import *
from models.pytorch_mssim import SSIM, RSSSIM, SpeedupRSSSIM
from loss.stn_ssim import STSSIM
from models.loss_function import ShiftedLoss
import matplotlib.pyplot as plt
import matplotlib
import tqdm

model_dict = {
    "RFNet": ResidualFeatureNet(),
    "STResNet_R": STResNet_R(),
    "STResNet_S": STResNet_S(),
    "STResNetRelu_R": STResNetRelu_R(),
    "STResNetRelu_S": STResNetRelu_S(),
    "STResNet16_R": STResNet16_R(),
    "STResNet16_S": STResNet16_S(),
    "STNResRFNet3v216": STNResRFNet3v216(),
    "ResNet": ResNet()
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
        chuncks = 100
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


def two_session(session1, session2, image_size, options, inference, loss_model, gpu_num):
    subs_session1 = subfolders(session1, preserve_prefix=True)
    subs_session1 = sorted(subs_session1)
    subs_session2 = subfolders(session2, preserve_prefix=True)
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
        im_gallery.sort()
        im_probe = subimages(probe, preserve_prefix=True)
        im_probe.sort()

        nim_gallery = len(im_gallery)
        nim_probe = len(im_probe)
        if nims == -1:
            nims = nim_gallery
            assert (nims == nim_probe)  # Check if image numbers in probe equals number in gallery
        else:
            assert (nims == nim_gallery and nims == nim_probe)  # Check for each folder

        probe_fv = calc_feats_more(*im_probe, size=image_size, options=options, model=inference, gpu_num=gpu_num)
        gallery_fv = calc_feats_more(*im_gallery, size=image_size, options=options, model=inference, gpu_num=gpu_num)

        feats_probe.append(probe_fv)
        feats_gallery.append(gallery_fv)

    feats_probe = torch.from_numpy(np.concatenate(feats_probe, 0)).cuda(gpu_num)
    feats_gallery = np.concatenate(feats_gallery, 0)
    feats_gallery2 = np.concatenate((feats_gallery, feats_gallery), 0)
    feats_gallery = torch.from_numpy(feats_gallery2).cuda(gpu_num)

    nl = nsubs * nims
    matching_matrix = np.ones((nl, nl)) * 1000000
    for i in range(nl):
        x = feats_probe
        y = feats_gallery[i: i + nl, :, :, :]
        bs, ch, he, wi = x.shape
        loss = np.ones(bs, ) * 1e5
        chuncks = 100
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
            loss = _loss(feats_probe, feats_gallery[i: i + nl, :, :, :], loss_model=loss_model)
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
        g_scores.append(float(np.min(matching_matrix[i, start_idx * nims: start_idx * nims + nims])))
        select = list(range(nl))
        for j in range(nims):
            select.remove(start_idx * nims + j)
        i_scores += list(np.min(np.reshape(matching_matrix[i, select], (-1, nims)), axis=1))
        print("[*] Processing genuine imposter for {} / {} \r".format(i, nsubs * nims))
        # sys.stdout.write("[*] Processing genuine imposter for {} / {} \r".format(i, nsubs * nims))
        # sys.stdout.flush()
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

    return np.array(g_scores), np.array(i_scores), matching_matrix


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
    parser.add_argument("--session2", type=str,
                        default="/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV3/GUI_Segment_Rotate/Session_2/",
                        dest="session2")
    parser.add_argument("--hyper_parameter", type=str,
                        default="../checkpoint/Joint-Finger-RFNet/MaskL_STResNetRelu_R_quadruplet_ssim_01-05-17-22-31/hyper_parameter.txt",
                        dest="hyper_parameter")
    parser.add_argument("--check_point", type=str, default="3000.pth", dest="check_point")
    parser.add_argument("--protocol", type=str, default="all_to_all", dest="protocol")
    parser.add_argument("--option", type=str, dest="option", default='RGB')
    parser.add_argument("--save_mmat", type=bool, dest="save_mmat", default=True)
    parser.add_argument("--gpu_num", type=int, dest="gpu_num", default=0)
    parser.add_argument("--if_draw", type=bool, dest="if_draw", default=True)
    args = parser.parse_args()

    para_dict = {}
    with open(args.hyper_parameter, "r") as para_file:
        para = para_file.readlines()
        for p in para:
            key = p.split(":")[0]
            value = p.split(":")[-1].strip('\n')
            if key == "input_size":
                v_tuple = (int(value[1:4]), int(value[6:9]))
                para_dict[key] = (128, 128)
            else:
                para_dict[key] = value

    # para_dict['data_range'] = "1.0"
    cls_num = ['01', '02', '04', '07']
    # cls_num = ['01']

    if not os.path.exists('.' + para_dict['checkpoint_dir'] + '/output'):
        os.mkdir('.' + para_dict['checkpoint_dir'] + '/output')

    inference = model_dict[para_dict['model']].cuda(args.gpu_num)
    inference.load_state_dict(torch.load('.' + para_dict['checkpoint_dir'] + '/ckpt_epoch_' + args.check_point))
    inference.eval()

    if para_dict['loss_type'] == "ssim":
        Loss = SSIM(data_range=float(para_dict['data_range']), size_average=False,
                    win_size=int(para_dict['win_size']), channel=int(para_dict['out_channel']))
    elif para_dict['loss_type'] == "rsssim":
        Loss = RSSSIM(data_range=float(para_dict['data_range']), size_average=False,
                      win_size=int(para_dict['win_size']), channel=int(para_dict['out_channel']),
                      v_shift=int(para_dict['vertical_size']),
                      h_shift=int(para_dict['horizontal_size']), angle=int(para_dict['rotate_angle']),
                      step=int(para_dict['step_size']))
    else:
        if para_dict['loss_type'] == "stssim":
            Loss = STSSIM(data_range=float(para_dict['data_range']), size_average=False,
                          win_size=int(para_dict['win_size']), channel=int(para_dict['out_channel']))
        elif para_dict['loss_type'] == "rsssim_speed":
            Loss = SpeedupRSSSIM(data_range=float(para_dict['data_range']), size_average=False,
                                 win_size=int(para_dict['win_size']), channel=int(para_dict['out_channel']),
                                 v_shift=int(para_dict['vertical_size']),
                                 h_shift=int(para_dict['horizontal_size']), angle=int(para_dict['rotate_angle']),
                                 step=int(para_dict['step_size']))
        else:
            if para_dict['loss_type'] == "shiftedloss":
                Loss = ShiftedLoss(hshift=int(para_dict['horizontal_size']), vshift=int(para_dict['vertical_size']))
    Loss.cuda(args.gpu_num)
    if para_dict['loss_type'] == "stssim":
        Loss.load_state_dict(torch.load('.' + para_dict['checkpoint_dir'] + '/loss_epoch_' + args.check_point))
    Loss.eval()

    for c in cls_num:
        test_path = os.path.join(args.test_path, c)
        session2 = os.path.join(args.session2, c)
        out_path = os.path.join('.' + para_dict['checkpoint_dir'] + '/output', c + ".mat")

        if args.protocol == "all_to_all":
            gscores, iscores, mmat = genuine_imposter_upright(test_path=test_path, image_size=para_dict['input_size'],
                                                              options=args.option, inference=inference, loss_model=Loss,
                                                              gpu_num=args.gpu_num)
        else:
            gscores, iscores, mmat = two_session(session1=test_path, session2=session2,
                                                 image_size=para_dict['input_size'],
                                                 options=args.option, inference=inference, loss_model=Loss,
                                                 gpu_num=args.gpu_num)

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
            src_mat.append(os.path.join('.' + para_dict['checkpoint_dir'] + '/output', c + ".mat"))
            label.append(c)

        draw_roc(src_mat, color, label, dst='.' + para_dict['checkpoint_dir'] + '/output')
