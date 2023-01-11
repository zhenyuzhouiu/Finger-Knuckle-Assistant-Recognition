import os
import shutil
import cv2
from PIL import Image

src_path = r"G:\finger_knuckle_2018\finger_knuckle_2018\FingerKnukcleDatabase\RAWr\Session-1"
dst_path = r"G:\finger_knuckle_2018\finger_knuckle_2018\FingerKnukcleDatabase\RAWr\Session-1-Cls-500"

if not os.path.exists(dst_path):
    os.mkdir(dst_path)


subject_name = os.listdir(src_path)
num_img = 0
for s in subject_name:
    subject_path = os.path.join(src_path, s)
    img_name = os.listdir(subject_path)
    for i in img_name:
        if i.split('-')[1] == "1":
            # src_file = os.path.join(src_path, f)
            # dst_file = os.path.join(dst_path, f)
            # img = cv2.imread(src_file)
            # image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # image.save(dst_file, dpi=(500.0, 500.0))
            subject, session, cls, sample = i.split("-")
            cls_path = os.path.join(dst_path, cls)
            if sample.split(".")[-1] == "bmp":
                if not os.path.exists(cls_path):
                    os.mkdir(cls_path)
                # sub_path = os.path.join(cls_path, subject)
                # if not os.path.exists(sub_path):
                #     os.mkdir(sub_path)
                src_file = os.path.join(subject_path, i)
                dst_file = os.path.join(cls_path, i.split(".")[0]+'.jpg')
                img = cv2.imread(src_file)
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                image.save(dst_file, dpi=(500.0, 500.0))
                # shutil.copy(src_file, dst_file)
                num_img += 1

print("Num img: " + str(num_img))
