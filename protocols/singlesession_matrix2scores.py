import sys
import numpy as np
import math
import argparse
from protocols.confusionmatrix.protocol_util import *
from inspect import getsourcefile
from scipy import io


current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)


def genuine_imposter(matrix_path, nsubs, nims):

    matt = io.loadmat(matrix_path)
    matt = matt['matt']
    g_scores = []
    i_scores = []
    for i in range(nsubs * nims):
        start_idx = int(math.floor(i / nims))
        start_remainder = int(i % nims)

        argmin_idx = np.argmin(matt[i, start_idx * nims: start_idx * nims + nims])
        g_scores.append(float(matt[i, start_idx * nims + argmin_idx]))
        select = list(range(nsubs * nims))
        # remove genuine matching score
        for j in range(nims):
            select.remove(start_idx * nims + j)
        # remove imposter matching scores of same index sample on other subjects
        for j in range(nsubs):
            if j == start_idx:
                continue
            select.remove(j * nims + start_remainder)
        i_scores += list(np.min(np.reshape(matt[i, select], (-1, nims - 1)), axis=1))
        print("[*] Processing genuine imposter for {} / {} \r".format(i, nsubs * nims))
        # sys.stdout.write("[*] Processing genuine imposter for {} / {} \r".format(i, nsubs * nims))
        # sys.stdout.flush()
    print("\n [*] Done")
    return np.array(g_scores), np.array(i_scores), matt


parser = argparse.ArgumentParser()
parser.add_argument("--matrix_path", type=str,
                    default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet-fkv3-session2-crossdb-thu/matt.mat",
                    dest="matrix_path")
parser.add_argument("--out_path", type=str,
                    default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet-fkv3-session2-crossdb-thu/protocol.npy",
                    dest="out_path")

parser.add_argument("--save_mmat", type=bool, dest="save_mmat", default=True)

args = parser.parse_args()

gscores, iscores, mmat = genuine_imposter(args.matrix_path, nsubs=610, nims=4)
if args.save_mmat:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores, "mmat": mmat})
else:
    np.save(args.out_path, {"g_scores": gscores, "i_scores": iscores})
