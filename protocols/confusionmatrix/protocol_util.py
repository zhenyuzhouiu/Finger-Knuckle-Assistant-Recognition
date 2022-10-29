# =========================================================
# @ Function File: Some util functions
# =========================================================

import os
from os.path import join


def mkdir_if_not_exists(*args):
    for arg in args:
        if not os.path.exists(arg):
            os.mkdir(arg)


def subfolders(src, preserve_prefix=False):
    if preserve_prefix:
        return [join(src, d) for d in os.listdir(src) if "." not in d]
    else:
        return [d for d in os.listdir(src) if "." not in d]


def subimages(src, preserve_prefix=False, ext=["JPG", "bmp", "jpg"]):
    def _hasext(f):
        for ext_ in ext:
            if ext_ in f:
                return True
        return False

    if preserve_prefix:
        return [join(src, f) for f in os.listdir(src) if _hasext(f)]
    else:
        return [f for f in os.listdir(src) if ext in f]
