import os
import cv2
import shutil

data_path = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/mask-seg-128/"

dst_path_test = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/mask-seg-100-70/"
dst_path_train = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/mask-seg-80-48/"

if os.path.exists(dst_path_test):
    shutil.rmtree(dst_path_test)
os.mkdir(dst_path_test)
if os.path.exists(dst_path_train):
    shutil.rmtree(dst_path_train)
os.mkdir(dst_path_train)

cls_name = os.listdir(data_path)
for c in cls_name:
    cls_path = os.path.join(data_path, c)
    test_cls = os.path.join(dst_path_test, c)
    os.mkdir(test_cls)
    train_cls = os.path.join(dst_path_train, c)
    os.mkdir(train_cls)
    sub_name = os.listdir(cls_path)
    for s in sub_name:
        sub_path = os.path.join(cls_path, s)
        test_sub = os.path.join(test_cls, s)
        os.mkdir(test_sub)
        train_sub = os.path.join(train_cls, s)
        os.mkdir(train_sub)
        img_name = os.listdir(sub_path)
        for i in img_name:
            img_path = os.path.join(sub_path, i)
            test_img = os.path.join(test_sub, i)
            train_img = os.path.join(train_sub, i)

            image = cv2.imread(img_path)
            # step2:-> keep ration 128x128->128x90
            h, w, _ = image.shape
            center_w = int(w / 2)
            center_h = int(h / 2)
            keep_ration_h = int((w * 70) / 100)
            crop_image = image[int((h - keep_ration_h) / 2):int((h - keep_ration_h) / 2) + keep_ration_h, 0:w]
            image_100_70 = cv2.resize(crop_image, (100, 70))
            cv2.imwrite(test_img, image_100_70)

            # step3:-> generate the 80x48 training image
            image_80_64 = image_100_70[10:58, 9:89]
            cv2.imwrite(train_img, image_80_64)
