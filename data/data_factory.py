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

from os.path import join, exists
from PIL import Image


def load_image(path, options='RGB', size=(128, 128)):
    assert (options in ["RGB", "L"])
    # the torch.ToTensor will scaling the [0, 255] to [0.0, 1.0]
    # if the numpy.ndarray has dtype = np.uint8
    image = np.array(Image.open(path).convert(options).resize(size=size), dtype=np.uint8)
    return image


def randpick_list(src, list_except=None):
    if not list_except:
        return src[np.random.randint(len(src))]
    else:
        src_cp = list(src)
        for exc in list_except:
            src_cp.remove(exc)
        return src_cp[np.random.randint(len(src_cp))]


class Factory(torch.utils.data.Dataset):
    def __init__(self, data_path, input_size=(128, 128), transform=None, valid_ext=['.jpg', '.bmp', '.png'], train=True):
        self.ext = valid_ext
        self.transform = transform
        self._has_ext = lambda f: True if [e for e in self.ext if e in f] else False
        self.folder = data_path
        self.input_size = input_size
        self.train = train

        if not exists(self.folder):
            raise RuntimeError('Dataset not found: {}'.format(self.folder))

        self.subfolder_names = [d for d in os.listdir(self.folder) if '.' not in d]
        if not self.subfolder_names:
            raise RuntimeError('Dataset must have subfolders indicating labels: {}'.format(self.folder))

        self._build_fdict()

    def __getitem__(self, index):
        if self.train:
            return self._triplet_trainitems(index)
        else:
            return self._get_testitems(index)

    def __len__(self):
        if self.train:
            return len(self.subfolder_names)
        else:
            return len(self.inames)

    def _build_fdict(self):
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

    def _get_testitems(self, index):
        fname = self.inames[index]
        labels = int(os.path.basename(os.path.abspath(join(fname, os.path.pardir))))
        img = load_image(fname)
        if self.transform is not None:
            img = self.transform(img)
        return img, labels
