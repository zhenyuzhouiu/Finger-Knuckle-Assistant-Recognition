# =========================================================
# @ DataLoader File: data_factory.py
#
# @ Target dataset: All the formatted dataset
#
# @ Notes: There should be some folders named [X, XX, ..]
#   under the test path, each of which contains several
#   images with valid extension (see Line 33). The number
#   of images doesn't need to be the same. As for triplet
#   selection, please read carefully from Line 76 to Line 95
# =========================================================

import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from os.path import join, exists
from PIL import Image
from torchvision.ops import roi_pool, roi_align
from data.augmentations import random_perspective, augment_hsv
from torchvision import transforms
from torch.utils.data import DataLoader


def load_image(path, options='RGB', size=(208, 184)):
    assert (options in ["RGB", "L"])
    # the torch.ToTensor will scaling the [0, 255] to [0.0, 1.0]
    # if the numpy.ndarray has dtype = np.uint8
    src_image = np.array(Image.open(path).convert(options), dtype=np.uint8)
    image = src_image[8:-8, :, :]
    # ration = h/w; size(w, h)
    ratio = size[1] / size[0]
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

    return resize_image


def load_feature_8(path, image_name):
    """
    1). load finger knuckle feature maps from yolov5
    2). for keeping the same feature map size, we use the roi_align

    return pooled_8.type:-> tensor; pooled_8.shape:-> [1, 320, 32, 32]
    """
    image_name = image_name.split('.')[0]
    # feature_8.shape:-> h*w*320
    feature_8 = np.load(join(path, image_name + '-8.npy'))
    h = feature_8.shape[0]
    w = feature_8.shape[1]
    # h*w*320 -> 320*h*w -> 1*320*h*w
    feature_8 = torch.from_numpy(np.expand_dims(np.transpose(feature_8, axes=(2, 0, 1)), axis=0)).float()
    boxes = torch.tensor([[0, 0, 0, w - 1, h - 1]]).float()
    # output_size (height, width).
    pooled_8 = roi_align(feature_8, boxes, [22, 26])
    pooled_8 = pooled_8.squeeze(0)
    norm_pooled_8 = pooled_8
    # for i in range(pooled_8.size(0)):
    #     pooled_i = pooled_8[i, :, :]
    #     min = pooled_i.min()
    #     max = pooled_i.max()
    #     pooled_i = (pooled_i - min)/(max - min)
    #     norm_pooled_8[i, :, :] = pooled_i.unsqueeze(0)

    return norm_pooled_8


def load_feature_16(path, image_name):
    """
    1). load finger knuckle feature maps from yolov5
    2). for keeping the same feature map size, we use the roi_align
    """
    image_name = image_name.split('.')[0]
    # feature_16.shape:-> h*w*640
    feature_16 = np.load(join(path, image_name + '-16.npy'))
    h = feature_16.shape[0]
    w = feature_16.shape[1]
    # h*w*640 -> 640*h*w -> 1*640*h*w
    feature_16 = torch.from_numpy(np.expand_dims(np.transpose(feature_16, axes=(2, 0, 1)), axis=0)).float()
    boxes = torch.tensor([[0, 0, 0, w - 1, h - 1]]).float()
    pooled_16 = roi_align(feature_16, boxes, [12, 14])
    pooled_16 = pooled_16.squeeze(0)
    norm_pooled_16 = pooled_16
    # for i in range(pooled_16.size(0)):
    #     pooled_i = pooled_16[i, :, :]
    #     min = pooled_i.min()
    #     max = pooled_i.max()
    #     pooled_i = (pooled_i - min) / (max - min)
    #     norm_pooled_16[i, :, :] = pooled_i.unsqueeze(0)

    return norm_pooled_16


def load_feature_32(path, image_name):
    """
    1). load finger knuckle feature maps from yolov5
    2). for keeping the same feature map size, we use the roi_align
    """
    image_name = image_name.split('.')[0]
    # feature_32.shape:-> h*w*1280
    feature_32 = np.load(join(path, image_name + '-32.npy'))
    h = feature_32.shape[0]
    w = feature_32.shape[1]
    # h*w*1280 -> 1280*h*w -> 1*1280*h*w
    feature_32 = torch.from_numpy(np.expand_dims(np.transpose(feature_32, axes=(2, 0, 1)), axis=0)).float()
    boxes = torch.tensor([[0, 0, 0, w - 1, h - 1]]).float()
    pooled_32 = roi_align(feature_32, boxes, [7, 8])
    pooled_32 = pooled_32.squeeze(0)
    norm_pooled_32 = pooled_32
    # for i in range(pooled_32.size(0)):
    #     pooled_i = pooled_32[i, :, :]
    #     min = pooled_i.min()
    #     max = pooled_i.max()
    #     pooled_i = (pooled_i - min) / (max - min)
    #     norm_pooled_32[i, :, :] = pooled_i.unsqueeze(0)

    return norm_pooled_32


def randpick_list(src, list_except=None):
    if not list_except:
        return src[np.random.randint(len(src))]
    else:
        src_cp = list(src)
        for exc in list_except:
            src_cp.remove(exc)
        return src_cp[np.random.randint(len(src_cp))]


def reminderpick_list(src, list_except=None):
    if not list_except:
        return src
    else:
        src_cp = list(src)
        for exc in list_except:
            src_cp.remove(exc)
        return src_cp


class Factory(torch.utils.data.Dataset):
    def __init__(self, img_path, feature_path, input_size,
                 transform=None, valid_ext=['.jpg', '.bmp', '.png'], train=True, n_tuple="triplet", if_augment=False):
        self.ext = valid_ext
        self.transform = transform
        self._has_ext = lambda f: True if [e for e in self.ext if e in f] else False
        self.folder = img_path
        self.feature_folder = feature_path
        self.train = train
        # input_size:-> (w, h)
        self.input_size = input_size
        self.n_tuple = n_tuple
        self.if_augment = if_augment

        if not exists(self.folder):
            raise RuntimeError('Dataset not found: {}'.format(self.folder))

        self.subfolder_names = [d for d in os.listdir(self.folder) if '.' not in d]
        if not self.subfolder_names:
            raise RuntimeError('Dataset must have subfolders indicating labels: {}'.format(self.folder))
        # get subject folder dictionary
        # get each image path
        self._build_fdict()

    def __getitem__(self, index):
        if self.train:
            if self.n_tuple == "feature":
                return self._feature_trainitems(index)
            elif self.n_tuple == "quadruplet":
                return self._quadruplet_trainitems(index)
            else:
                return self._triplet_trainitems(index)
        else:
            return self._get_testitems(index)

    def __len__(self):
        if self.train:
            return len(self.subfolder_names)
        else:
            return len(self.inames)

    def _build_fdict(self):
        '''
        self.fdict:-> dictionary {subject_folder: image_name}
        self.inames contains each image absolute path self.folder + sf + f
        '''
        self.fdict = {}
        self.inames = []
        self.min_subj = 1000000
        for sf in self.subfolder_names:
            inames = [d for d in os.listdir(join(self.folder, sf)) if self._has_ext(d)]
            if len(inames) < 1 and self.train:
                raise RuntimeError('Pls make sure there are at least one images in {}'.format(
                    join(self.folder, sf)
                ))
            self.inames = self.inames + [join(self.folder, sf, f) for f in inames]
            self.fdict[sf] = inames
            if self.min_subj > len(inames):
                self.min_subj = len(inames)

    def _feature_dict_(self):
        self.feature_fdict = {}
        self.feature_fnames = []
        self.min_subj = 1000000
        for sf in self.subfolder_names:
            fnames = [d for d in os.listdir(join(self.feature_folder, sf))]
            if len(fnames) < 1 and self.train:
                raise RuntimeError('Pls make sure there are at least one feature in {}'.format(
                    join(self.feature_folder, sf)
                ))
            self.feature_fnames = self.feature_fnames + [join(self.feature_folder, sf, f) for f in fnames]
            self.feature_fdict[sf] = fnames
            if self.min_subj > len(fnames):
                self.min_subj = len(fnames)

    def new_triplet_trainitems(self, index):
        # ======================= get images and corresponding features and label
        # Per index, per subject
        selected_folder = self.subfolder_names[index]
        list_folders = [selected_folder]
        anchor = randpick_list(self.fdict[selected_folder])
        positive = reminderpick_list(self.fdict[selected_folder], [anchor])

        img = [load_image(join(self.folder, selected_folder, anchor), options='RGB', size=self.input_size)]
        for p in positive:
            img.append(load_image(join(self.folder, selected_folder, p), options='RGB', size=self.input_size))
        for i in range(2):
            negative_folder = randpick_list(self.subfolder_names, list_folders)
            list_folders.append(negative_folder)
            negative = reminderpick_list(self.fdict[negative_folder])
            for n in negative:
                img.append(load_image(join(self.folder, negative_folder, n), options='RGB', size=self.input_size))

        # img is the data
        # junk is the label
        img = np.concatenate(img, axis=-1)
        junk = np.array([0])
        if self.transform is not None:
            # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
            # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            # if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
            # or if the numpy.ndarray has dtype = np.uint8
            # In the other cases, tensors are returned without normalization.
            img = self.transform(img)

        return img, junk

    def _triplet_trainitems(self, index):
        # Per index, per subject
        # Negative samples 5 times than positive

        selected_folder = self.subfolder_names[index]
        anchor = randpick_list(self.fdict[selected_folder])
        positive = randpick_list(self.fdict[selected_folder], [anchor])

        img = []
        # options = 'L' just convert image to gray image
        # img.append(np.expand_dims(load_image(join(self.folder, selected_folder, positive), options='L'), -1))
        # img.append(np.expand_dims(load_image(join(self.folder, selected_folder, anchor), options='L'), -1))
        img.append(load_image(join(self.folder, selected_folder, anchor), options='RGB', size=self.input_size))
        img.append(load_image(join(self.folder, selected_folder, positive), options='RGB', size=self.input_size))

        for i in range(5):
            negative_folder = randpick_list(self.subfolder_names, [selected_folder])
            negative = randpick_list(self.fdict[negative_folder])
            # img.append(np.expand_dims(load_image(join(self.folder, negative_folder, negative), options='RGB'), -1))
            img.append(load_image(join(self.folder, negative_folder, negative), options='RGB', size=self.input_size))

        img = np.concatenate(img, axis=-1)
        junk = np.array([0])
        if self.transform is not None:
            # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
            # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            # if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
            # or if the numpy.ndarray has dtype = np.uint8
            # In the other cases, tensors are returned without scaling.
            img = self.transform(img)
        # img is the data
        # junk is the label
        return img, junk
    def _masktriplet_trainitems(self, index):
        # ======================= get images and corresponding features and label
        # Per index, per subject
        selected_folder = self.subfolder_names[index]
        list_folders = [selected_folder]
        anchor = randpick_list(self.fdict[selected_folder])
        positive = reminderpick_list(self.fdict[selected_folder], [anchor])

        # options = 'L' just convert image to gray image
        # img = []
        # img.append(np.expand_dims(load_image(join(self.folder, selected_folder, positive), options='L'), -1))
        # img.append(np.expand_dims(load_image(join(self.folder, selected_folder, anchor), options='L'), -1))
        if self.if_augment:
            src = load_image(join(self.folder, selected_folder, anchor), options='RGB', size=self.input_size)
            src = augment_hsv(src)
            src, ma = random_perspective(src)
            img = [src]
            mask = [ma]
        else:
            img = [load_image(join(self.folder, selected_folder, anchor), options='RGB', size=self.input_size)]
            # self.input_size:-> (w, h)
            ma = np.ones((1, int(self.input_size[1] / 4), int(self.input_size[0] / 4)), dtype=img[0].dtype)
            mask = [ma]
        for p in positive:
            if self.if_augment:
                src = load_image(join(self.folder, selected_folder, p), options='RGB', size=self.input_size)
                src = augment_hsv(src)
                src, ma = random_perspective(src)
                img.append(src)
                mask.append(ma)
            else:
                img.append(load_image(join(self.folder, selected_folder, p), options='RGB', size=self.input_size))
                ma = np.ones((1, int(self.input_size[1] / 4), int(self.input_size[0] / 4)), dtype=img[0].dtype)
                mask.append(ma)

        # Negative samples 2 times than positive
        for i in range(2):
            negative_folder = randpick_list(self.subfolder_names, list_folders)
            list_folders.append(negative_folder)
            negative = reminderpick_list(self.fdict[negative_folder])
            for n in negative:
                if self.if_augment:
                    src = load_image(join(self.folder, negative_folder, n), options='RGB', size=self.input_size)
                    src = augment_hsv(src)
                    src, ma = random_perspective(src)
                    img.append(src)
                    mask.append(ma)
                else:
                    img.append(load_image(join(self.folder, negative_folder, n), options='RGB', size=self.input_size))
                    ma = np.ones((1, int(self.input_size[1] / 4), int(self.input_size[0] / 4)), dtype=img[0].dtype)
                    mask.append(ma)

        # img is the data
        # junk is the label
        img = np.concatenate(img, axis=-1)
        junk = np.array([0])
        if self.transform is not None:
            # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
            # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            # if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
            # or if the numpy.ndarray has dtype = np.uint8
            # In the other cases, tensors are returned without normalization.
            img = self.transform(img)

        mask = np.concatenate(mask, axis=0)
        mask = (torch.from_numpy(mask)).type(img.dtype)

        return img, mask, junk

    def _feature_trainitems(self, index):
        # ======================= get images and corresponding features and label
        # Per index, per subject
        selected_folder = self.subfolder_names[index]
        list_folders = [selected_folder]
        anchor = randpick_list(self.fdict[selected_folder])
        positive = reminderpick_list(self.fdict[selected_folder], [anchor])

        # options = 'L' just convert image to gray image
        # img = []
        # img.append(np.expand_dims(load_image(join(self.folder, selected_folder, positive), options='L'), -1))
        # img.append(np.expand_dims(load_image(join(self.folder, selected_folder, anchor), options='L'), -1))
        img = [load_image(join(self.folder, selected_folder, anchor), options='RGB', size=self.input_size)]
        stride_8 = load_feature_8(join(self.feature_folder, selected_folder), image_name=anchor)
        stride_16 = load_feature_16(join(self.feature_folder, selected_folder), image_name=anchor)
        stride_32 = load_feature_32(join(self.feature_folder, selected_folder), image_name=anchor)
        for p in positive:
            img.append(load_image(join(self.folder, selected_folder, p), options='RGB', size=self.input_size))
            stride_8 = torch.cat([stride_8, load_feature_8(join(self.feature_folder, selected_folder), image_name=p)],
                                 dim=0)
            stride_16 = torch.cat(
                [stride_16, load_feature_16(join(self.feature_folder, selected_folder), image_name=p)], dim=0)
            stride_32 = torch.cat(
                [stride_32, load_feature_32(join(self.feature_folder, selected_folder), image_name=p)], dim=0)

        # Negative samples 2 times than positive
        for i in range(2):
            negative_folder = randpick_list(self.subfolder_names, list_folders)
            list_folders.append(negative_folder)
            negative = reminderpick_list(self.fdict[negative_folder])
            for n in negative:
                img.append(load_image(join(self.folder, negative_folder, n), options='RGB', size=self.input_size))
                stride_8 = torch.cat(
                    [stride_8, load_feature_8(join(self.feature_folder, negative_folder), image_name=n)], dim=0)
                stride_16 = torch.cat(
                    [stride_16, load_feature_16(join(self.feature_folder, negative_folder), image_name=n)], dim=0)
                stride_32 = torch.cat(
                    [stride_32, load_feature_32(join(self.feature_folder, negative_folder), image_name=n)], dim=0)
        # img is the data
        # junk is the label
        img = np.concatenate(img, axis=-1)
        junk = np.array([0])
        if self.transform is not None:
            # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
            # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            # if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
            # or if the numpy.ndarray has dtype = np.uint8
            # In the other cases, tensors are returned without scaling.
            img = self.transform(img)

        # stride_8.shape:-> [320*3*samples_subject, 32, 32]; stride_8.type:-> tensor
        # stride_16.shape:-> [640*3*samples_subject, 16, 16]
        # stride_32.shape:-> [1280*3*samples_subject, 8, 8]
        return img, junk, stride_8, stride_16, stride_32

    def _quadruplet_trainitems(self, index):
        # ======================= get images and corresponding features and label
        # Per index, per subject
        selected_folder = self.subfolder_names[index]
        list_folders = [selected_folder]
        anchor = randpick_list(self.fdict[selected_folder])
        positive = reminderpick_list(self.fdict[selected_folder], [anchor])

        img = [load_image(join(self.folder, selected_folder, anchor), options='RGB', size=self.input_size)]
        for p in positive:
            img.append(load_image(join(self.folder, selected_folder, p), options='RGB', size=self.input_size))

        # Negative samples 2 times than positive
        # the first class negative sample
        for i in range(1):
            negative_folder = randpick_list(self.subfolder_names, list_folders)
            list_folders.append(negative_folder)
            negative = reminderpick_list(self.fdict[negative_folder])
            for n in negative:
                img.append(load_image(join(self.folder, negative_folder, n), options='RGB', size=self.input_size))

        for i in range(1):
            negative_folder = randpick_list(self.subfolder_names, list_folders)
            list_folders.append(negative_folder)
            negative = reminderpick_list(self.fdict[negative_folder])
            for n in negative:
                img.append(load_image(join(self.folder, negative_folder, n), options='RGB', size=self.input_size))

        # img is the data
        # junk is the label
        img = np.concatenate(img, axis=-1)
        junk = np.array([0])
        if self.transform is not None:
            # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
            # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            # if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
            # or if the numpy.ndarray has dtype = np.uint8
            # In the other cases, tensors are returned without scaling.
            img = self.transform(img)
        # img.shape = [samples_subject * 5 * 3, h, w]
        # img.shape = [75, h, w]
        return img, junk

    def _get_testitems(self, index):
        fname = self.inames[index]
        labels = int(os.path.basename(os.path.abspath(join(fname, os.path.pardir))))
        img = load_image(fname)
        if self.transform is not None:
            img = self.transform(img)
        return img, labels


if __name__ == "__main__":
    train_path = '/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/mask-seg/03/'
    feature_path = '/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/feature/03/'
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = Factory(train_path, feature_path, (128, 128), transform=transform,
                      valid_ext=['.bmp', '.jpg', '.JPG'], train=True, n_tuple='triplet', if_augment=True)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for i, (x, _) in enumerate(dataloader):
        print(i)
