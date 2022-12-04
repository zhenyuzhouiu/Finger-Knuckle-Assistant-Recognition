# ----------------------------------------------------------
# The code is to fill the missed fingerprint matching scores.

# from the fp_des_score matching score files, some fingerprint cannot be segmented
# result in incomplete matching score matrix.
# ----------------------------------------------------------

import os
import pandas as pd
import shutil

src_path = "/media/zhenyuzhou/Data/finger_knuckle_2018/5_fusion/fp_des_score/"
dst_path = "/media/zhenyuzhou/Data/finger_knuckle_2018/5_fusion/valid_fp_des_score/"
if os.path.exists(dst_path):
    shutil.rmtree(dst_path)
os.mkdir(dst_path)

files = os.listdir(src_path)
for f in files:
    cls = f.split(".")[0].split("_")[-1]
    csv_file = os.path.join(src_path, f)
    dst_csv_file = os.path.join(dst_path, f)
    # header=None read data from the first row (don's have header)
    df = pd.read_csv(csv_file, index_col=0)
    rows = len(df.index)
    cols = len(df.columns)
    if len(df.columns) == len(df.index) == 600:
        df.to_csv(dst_csv_file)
        print(f + " is complete matching score matrix!"+'\n')
        continue

    # insert column
    for sub in range(1, 121):
        for sam in range(1, 6):
            index = str(sub).zfill(3) + '-'+cls+'-' + str(sam)
            i_loc = (sub - 1) * 5 + sam - 1
            if index not in df.columns:
                # insert column into DataFrame
                df.insert(i_loc, index, ["1"]*rows, allow_duplicates=False)
                print(f+" don't have "+index+'\n')
    df = df.T
    rows = len(df.index)
    cols = len(df.columns)
    # insert row
    for sub in range(1, 121):
        for sam in range(1, 6):
            index = str(sub).zfill(3) + '-'+cls+'-' + str(sam)
            i_loc = (sub - 1) * 5 + sam - 1
            if index not in df.columns:
                # insert column into DataFrame
                df.insert(i_loc, index, ["1"]*rows, allow_duplicates=False)

    df = df.T
    df.to_csv(dst_csv_file)



