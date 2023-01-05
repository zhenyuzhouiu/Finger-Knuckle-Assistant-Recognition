import os
import shutil
import cv2


src_path = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV3/GUI_Segment"
dst_path = "/home/zhenyuzhou/Pictures/Finger-Knuckle-Database/PolyUKnuckleV3/GUI_Segment_Rotate"
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
            image = cv2.imread(img_path)
            rot_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            dst_img = os.path.join(dst_sub, i)
            cv2.imwrite(dst_img, rot_img)




