# ----------------------------------------------------------
# The code is to fill the missed fingerprint matching scores.

# from the fp_des_score matching score files, some fingerprint cannot be segmented
# result in incomplete matching score matrix.
# ----------------------------------------------------------

import os
import shutil
import cv2


def load_image(path, size=(128, 128)):
    src_image = cv2.imread(path)
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


src_path = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/mask-seg/"
dst_path = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/mask-seg-128/"
if os.path.exists(dst_path):
    shutil.rmtree(dst_path)
os.mkdir(dst_path)

cls_name = os.listdir(src_path)
for c in cls_name:
    cls_path = os.path.join(src_path, c)
    dst_cls = os.path.join(dst_path, c)
    os.mkdir(dst_cls)
    sub_name = os.listdir(cls_path)
    for s in sub_name:
        sub_path = os.path.join(cls_path, s)
        dst_sub = os.path.join(dst_cls, s)
        os.mkdir(dst_sub)
        img_name = os.listdir(sub_path)
        for i in img_name:
            img_path = os.path.join(sub_path, i)
            resize_img = load_image(img_path, size=(128, 128))
            dst_img = os.path.join(dst_sub, i)
            cv2.imwrite(dst_img, resize_img)




