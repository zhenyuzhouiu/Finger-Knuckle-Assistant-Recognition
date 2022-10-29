# =========================================================
# @ Protocol File: Two sessions (Probe / Gallery)
#
# @ Target dataset:
# @ Parameter Settings:
#       save_mmat:  whether save matching matrix or not,
#                   could be helpful for plot CMC
#
# @ Notes:  also could be used on PolyU 2D if format the
#           dataset like "session1" and "session2" under
#           PolyU 2D data folder
# =========================================================

from __future__ import division
import os
from scipy import io
import sys
from PIL import Image
import numpy as np
import torch
import math
import argparse
from torch.autograd import Variable

from inspect import getsourcefile

current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def genuine_imposter(matt_path, nsubs, nims):
    matt = io.loadmat(matt_path)
    matching_matrix = matt['matching_matrix']

    nl = nsubs * nims

    for i in range(1, nl):
        tmp = matching_matrix[i, -i:].copy()
        matching_matrix[i, i:] = matching_matrix[i, :-i]
        matching_matrix[i, :i] = tmp
    print("\n [*] Done")

    g_scores = []
    i_scores = []
    for i in range(nl):
        start_idx = int(math.floor(i / nims))
        start_remainder = int(i % nims)
        g_scores.append(float(np.min(matching_matrix[i, start_idx * nims: start_idx * nims + nims])))
        select = list(range(nl))
        for j in range(nims):
            select.remove(start_idx * nims + j)
        i_scores += list(np.min(np.reshape(matching_matrix[i, select], (-1, nims)), axis=1))
        print("[*] Processing genuine imposter for {} / {} \r".format(i, nsubs * nims))
        # sys.stdout.write("[*] Processing genuine imposter for {} / {} \r".format(i, nsubs * nims))
        # sys.stdout.flush()
    print("\n [*] Done")
    return np.array(g_scores), np.array(i_scores), matching_matrix


parser = argparse.ArgumentParser()
parser.add_argument("--matt_path", type=str,
                    default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet-fkv3_session1_105_221-two-session/matching_matrix.mat",
                    dest="matt_path")
parser.add_argument("--out_path", type=str,
                    default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet-fkv3_session1_105_221-two-session/fknet.npy",
                    dest="out_path")
parser.add_argument("--save_mmat", type=bool, dest="save_mmat", default=True)

args = parser.parse_args()

gscores, iscores, mmat = genuine_imposter(args.matt_path, nsubs=104, nims=6)
if args.save_mmat:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores, "mmat": mmat})
else:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores})
