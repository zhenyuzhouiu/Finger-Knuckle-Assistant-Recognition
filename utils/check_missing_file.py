import os
import shutil


def missed_iso(src_path, dst_path, save_path):
    fp_image = os.listdir(src_path)
    iso_file = os.listdir(dst_path)
    missed_num = 0
    for f in fp_image:
        if f.split(".")[0]+'.ist' not in iso_file:
            src_file = os.path.join(src_path, f)
            save_file = os.path.join(save_path, f)
            shutil.copy(src_file, save_file)
            missed_num += 1
    print("Missed Num: " + str(missed_num))


def missed_segment(src_path, dst_path, save_path):
    raw_image = os.listdir(src_path)
    fp_image = os.listdir(dst_path)
    one_sample = fp_image[0]
    subject, slap, cls, sample = one_sample.split("-")
    missed_num = 0
    for r_i in raw_image:
        r = r_i.split(".")[0]
        subject_id = r.split("-")[0]
        sample_id = r.split("-")[-1]
        src_name = subject_id + "-" + slap + "-" + cls + "-" + sample_id + '.jpg'
        if src_name not in fp_image:
            src_file = os.path.join(src_path, r_i)
            save_file = os.path.join(save_path, r_i)
            shutil.copy(src_file, save_file)
            missed_num += 1
    print("Missed Num: " + str(missed_num))


if __name__ == "__main__":
    src_path = r"G:\finger_knuckle_2018\finger_knuckle_2018\FingerKnukcleDatabase\RAWr\Session-1-Cls-500\13"
    dst_path = r"G:\finger_knuckle_2018\finger_knuckle_2018\FingerKnukcleDatabase\SEG\Session-1-updated-Cls-500\10"
    save_path = r"G:\finger_knuckle_2018\finger_knuckle_2018\FingerKnukcleDatabase\SEG\Session-1-updated-Cls-500\10-segment-missed"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # missed_iso(src_path, dst_path, save_path)
    missed_segment(src_path, dst_path, save_path)


