import os
import shutil

src_path = r"F:\finger_knuckle_2018\finger_knuckle_2018\4_FPscore\FPDatabase"
dst_path = r"F:\finger_knuckle_2018\finger_knuckle_2018\4_FPscore\FPClass"

if not os.path.exists(dst_path):
    os.mkdir(dst_path)


file_name = os.listdir(src_path)
num_img = 0
for f in file_name:
    subject, session, cls, sample = f.split("-")
    cls_path = os.path.join(dst_path, cls)
    if sample.split(".")[-1] == "jpg":
        if not os.path.exists(cls_path):
            os.mkdir(cls_path)
        sub_path = os.path.join(cls_path, subject)
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)
        src_file = os.path.join(src_path, f)
        dst_file = os.path.join(sub_path, f)
        shutil.copy(src_file, dst_file)
        num_img += 1

print("Num img: " + str(num_img))
