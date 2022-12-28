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
    if options == 'RGB':
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
    else:
        image = src_image[8:-8, :]
        # ration = h/w; size(w, h)
        ratio = size[1] / size[0]
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

    return resize_image


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
    def __init__(self, img_path, input_size,
                 transform=None, valid_ext=['.jpg', '.bmp', '.png'], train=True, n_tuple="triplet",
                 if_aug=False, if_hsv=False, if_rotation=False, if_translation=False, if_scale=False):
        self.ext = valid_ext
        self.transform = transform
        self._has_ext = lambda f: True if [e for e in self.ext if e in f] else False
        self.folder = img_path
        self.train = train
        # input_size:-> (w, h)
        self.input_size = input_size
        self.n_tuple = n_tuple
        self.if_aug = if_aug
        self.if_hsv = if_hsv
        self.if_rotation = if_rotation
        self.if_translation = if_translation
        self.if_scale = if_scale

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
            if self.n_tuple == "oldtriplet":
                return self._oldtriplet_trainitems(index)
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

    def _oldtriplet_trainitems(self, index):
        # Per index, per subject
        # Negative samples 5 times than positive

        selected_folder = self.subfolder_names[index]
        anchor = randpick_list(self.fdict[selected_folder])
        positive = randpick_list(self.fdict[selected_folder], [anchor])

        img = []
        img.append(
            np.expand_dims(load_image(join(self.folder, selected_folder, positive), options='L', size=self.input_size),
                           -1))
        img.append(
            np.expand_dims(load_image(join(self.folder, selected_folder, anchor), options='L', size=self.input_size),
                           -1))
        for i in range(10):
            negative_folder = randpick_list(self.subfolder_names, [selected_folder])
            negative = randpick_list(self.fdict[negative_folder])
            img.append(np.expand_dims(
                load_image(join(self.folder, negative_folder, negative), options='L', size=self.input_size), -1))
        img = np.concatenate(img, axis=-1)
        junk = np.array([0])
        if self.transform is not None:
            img = self.transform(img)
        return img, junk

    def _triplet_trainitems(self, index):
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

    def _quadruplet_trainitems(self, index):
        # ======================= get images and corresponding features and label
        # Per index, per subject
        selected_folder = self.subfolder_names[index]
        list_folders = [selected_folder]
        anchor = randpick_list(self.fdict[selected_folder])
        positive = reminderpick_list(self.fdict[selected_folder], [anchor])

        if self.if_aug:
            src = load_image(join(self.folder, selected_folder, anchor), options='RGB', size=self.input_size)
            if self.if_hsv:
                src = augment_hsv(src)
            if self.if_rotation:
                src = random_perspective(src, degrees=5, translate=0, scale=0)
            if self.if_translation:
                src = random_perspective(src, degrees=0, translate=0.1, scale=0)
            if self.if_scale:
                src = random_perspective(src, degrees=0, translate=0, scale=0.1)
            img = [src]
        else:
            img = [load_image(join(self.folder, selected_folder, anchor), options='RGB', size=self.input_size)]
        for p in positive:
            if self.if_aug:
                src = load_image(join(self.folder, selected_folder, p), options='RGB', size=self.input_size)
                if self.if_hsv:
                    src = augment_hsv(src)
                if self.if_rotation:
                    src = random_perspective(src, degrees=5, translate=0, scale=0)
                if self.if_translation:
                    src = random_perspective(src, degrees=0, translate=0.1, scale=0)
                if self.if_scale:
                    src = random_perspective(src, degrees=0, translate=0, scale=0.1)
                img.append(src)
            else:
                img.append(load_image(join(self.folder, selected_folder, p), options='RGB', size=self.input_size))

        # Negative samples 2 times than positive
        # the first class negative sample
        for i in range(1):
            negative_folder = randpick_list(self.subfolder_names, list_folders)
            list_folders.append(negative_folder)
            negative = reminderpick_list(self.fdict[negative_folder])
            for n in negative:
                if self.if_aug:
                    src = load_image(join(self.folder, negative_folder, n), options='RGB', size=self.input_size)
                    if self.if_hsv:
                        src = augment_hsv(src)
                    if self.if_rotation:
                        src = random_perspective(src, degrees=5, translate=0, scale=0)
                    if self.if_translation:
                        src = random_perspective(src, degrees=0, translate=0.1, scale=0)
                    if self.if_scale:
                        src = random_perspective(src, degrees=0, translate=0, scale=0.1)
                    img.append(src)
                else:
                    img.append(load_image(join(self.folder, negative_folder, n), options='RGB', size=self.input_size))

        for i in range(1):
            negative_folder = randpick_list(self.subfolder_names, list_folders)
            list_folders.append(negative_folder)
            negative = reminderpick_list(self.fdict[negative_folder])
            for n in negative:
                if self.if_aug:
                    src = load_image(join(self.folder, negative_folder, n), options='RGB', size=self.input_size)
                    if self.if_hsv:
                        src = augment_hsv(src)
                    if self.if_rotation:
                        src = random_perspective(src, degrees=5, translate=0, scale=0)
                    if self.if_translation:
                        src = random_perspective(src, degrees=0, translate=0.1, scale=0)
                    if self.if_scale:
                        src = random_perspective(src, degrees=0, translate=0, scale=0.1)
                    img.append(src)
                else:
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
