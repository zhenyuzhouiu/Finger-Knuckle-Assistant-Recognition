import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from protocols.plot.plotroc_basic import *
import scipy.io as io


def get_score(matching_matrix, subject_num=120, sample_num=5):
    feats_length = []
    nfeats = 0
    for i in range(subject_num):
        nfeats += sample_num
        feats_length.append(sample_num)
    feats_length = np.array(feats_length)
    acc_len = np.cumsum(feats_length)
    feats_start = acc_len - feats_length

    g_scores = []
    i_scores = []
    # nfeats: number of features
    for i in range(nfeats):
        # in case of multiple occurrences of the maximum values,
        # the indices corresponding to the first occurrence are returned.
        subj_idx = np.argmax(acc_len > i)
        g_select = [feats_start[subj_idx] + k for k in range(feats_length[subj_idx])]
        for i_i in range(i + 1):
            if i_i in g_select:
                g_select.remove(i_i)
        i_select = list(range(nfeats))
        # remove g_select
        for subj_i in range(subj_idx + 1):
            for k in range(feats_length[subj_i]):
                i_select.remove(feats_start[subj_i] + k)
        if len(g_select) != 0:
            g_scores += list(matching_matrix[i, g_select])
        if len(i_select) != 0:
            i_scores += list(matching_matrix[i, i_select])
    return np.array(g_scores), np.array(i_scores)


def draw_roc(g_scores, i_scores, save_path, label="Fingerprint", color="#ff0000"):
    x, y = calc_coordinates(g_scores, i_scores)
    print("[*] EER: {}".format(calc_eer(x, y)))
    EER = "%.3f%%" % (calc_eer(x, y) * 100)
    lines = plt.plot(x, y, label='ROC')
    plt.setp(lines, 'color', color, 'linewidth', 3, 'label', label + "; EER: " + str(EER))

    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)

    plt.grid(True)
    plt.xlabel(r'False Accept Rate', fontsize=18)
    plt.ylabel(r'Genuine Accept Rate', fontsize=18)
    legend = plt.legend(loc='lower right', shadow=False, prop={'size': 16})
    plt.xlim(xmin=np.min(x))
    plt.xlim(xmax=0)
    plt.ylim(ymax=1)
    plt.ylim(ymin=0.4)

    ax = plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)

    plt.xticks(np.array([-4, -2, 0]), ['$10^{-4}$', '$10^{-2}$', '$10^{0}$'], fontsize=16)
    plt.yticks(np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]), fontsize=16)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def get_scores_from_pd(fp_score):
    # ---------------------- read fp score and normalize
    # fp_score = fp_score.drop(columns=0).drop([0])
    fp_matrix = fp_score.values.tolist()
    fp_matrix = np.array(fp_matrix).astype(np.float64)
    fp_g_scores, fp_i_scores = get_score(fp_matrix)
    fp_g_valid, fp_i_valid = [], []
    for i in range(len(fp_g_scores)):
        if fp_g_scores[i] < 0:
            continue
        else:
            fp_g_valid.append(fp_g_scores[i])
    for i in range(len(fp_i_scores)):
        if fp_i_scores[i] < 0:
            continue
        else:
            fp_i_valid.append(fp_i_scores[i])
    fp_g_valid, fp_i_valid = np.array(fp_g_valid), np.array(fp_i_valid)
    g_min = np.min(fp_g_valid)
    g_max = np.max(fp_g_valid)
    i_min = np.min(fp_i_valid)
    i_max = np.max(fp_i_valid)
    max = np.max(np.array([g_max, i_max]))
    min = np.min(np.array([g_min, i_min]))
    fp_matrix = (max - fp_matrix) / (max - min)
    fp_g_scores, fp_i_scores = get_score(fp_matrix)
    fp_g_valid, fp_i_valid = [], []
    for i in range(len(fp_g_scores)):
        if fp_g_scores[i] > 1:
            fp_g_valid.append(1)
        else:
            fp_g_valid.append(fp_g_scores[i])
    for i in range(len(fp_i_scores)):
        if fp_i_scores[i] > 1:
            fp_i_valid.append(1)
        else:
            fp_i_valid.append(fp_i_scores[i])
    fp_g_scores, fp_i_scores = np.array(fp_g_valid), np.array(fp_i_valid)

    return fp_g_scores, fp_i_scores


# def fill_empty(pd_csv):
#     df = pd.read_csv(csv_file, index_col=0)
#     rows = len(df.index)
#     cols = len(df.columns)
#     if len(df.columns) == len(df.index) == 600:
#         df.to_csv(dst_csv_file)
#         print(f + " is complete matching score matrix!" + '\n')
#         continue
#
#     # insert column
#     for sub in range(1, 121):
#         for sam in range(1, 6):
#             index = str(sub).zfill(3) + '-' + cls + '-' + str(sam)
#             i_loc = (sub - 1) * 5 + sam - 1
#             if index not in df.columns:
#                 # insert column into DataFrame
#                 df.insert(i_loc, index, ["1"] * rows, allow_duplicates=False)
#                 print(f + " don't have " + index + '\n')
#     df = df.T
#     rows = len(df.index)
#     cols = len(df.columns)
#     # insert row
#     for sub in range(1, 121):
#         for sam in range(1, 6):
#             index = str(sub).zfill(3) + '-' + cls + '-' + str(sam)
#             i_loc = (sub - 1) * 5 + sam - 1
#             if index not in df.columns:
#                 # insert column into DataFrame
#                 df.insert(i_loc, index, ["1"] * rows, allow_duplicates=False)
#
#     df = df.T
#     df.to_csv(dst_csv_file)



if __name__ == "__main__":
    src_path = r"G:\finger_knuckle_2018\finger_knuckle_2018\FingerKnukcleDatabase\SEG\Session-1-updated-Cls-500\03-iso-verifinger\verifinger.txt"
    dst_path = r"G:\finger_knuckle_2018\finger_knuckle_2018\FingerKnukcleDatabase\SEG\Session-1-updated-Cls-500\fp_03.csv"
    roc_path = r"G:\finger_knuckle_2018\finger_knuckle_2018\FingerKnukcleDatabase\SEG\Session-1-updated-Cls-500\fp_03.pdf"
    mat_path = r"G:\finger_knuckle_2018\finger_knuckle_2018\FingerKnukcleDatabase\SEG\Session-1-updated-Cls-500\fp_03.mat"
    slap_knuckle = "1"
    cls_name = "03"

    col_name = []
    for i in range(1, 121):
        for j in range(1, 6):
            name = str(i).zfill(3) + "-" + slap_knuckle + "-" + cls_name + "-" + str(j)
            col_name.append(name)

    matrix = pd.DataFrame(columns=col_name, index=col_name, data=-1)

    print(matrix.head())

    num = 0
    dict_score = {}
    with open(src_path, "r") as score_file:
        lines = score_file.readlines()
        for l in lines:
            pair = l.split(":")[0]
            score = l.split(":")[-1].strip("\n")
            dict_score[pair] = score

    for i in range(1, 121):
        for j in range(1, 6):
            subject1 = str(i).zfill(3) + "-" + slap_knuckle + "-" + cls_name + "-" + str(j)
            for m in range(1, 121):
                for n in range(1, 6):
                    subject2 = str(m).zfill(3) + "-" + slap_knuckle + "-" + cls_name + "-" + str(n)
                    pair = subject1 + "_" + subject2
                    if pair in dict_score.keys():
                        matrix[subject1][subject2] = dict_score[pair]
                        num += 1
                        print(num)

    print(matrix.head())

    matrix.to_csv(dst_path)
    fp_g_scores, fp_i_scores = get_scores_from_pd(matrix)

    io.savemat(mat_path, {"g_scores": np.array(fp_g_scores), "i_scores": np.array(fp_i_scores)})

    draw_roc(fp_g_scores, fp_i_scores, roc_path)
