import os
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot.plotroc_basic import *
import matplotlib


def get_score(matching_matrix):
    feats_length = []
    nfeats = 0
    for i in range(120):
        nfeats += 5
        feats_length.append(5)
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


def draw_roc(scores, save_path, label, color):
    l_eer = []
    for i in range(len(scores)):
        score = scores[i]
        g_scores = np.array(score[0])
        i_scores = np.array(score[1])

        x, y = calc_coordinates(g_scores, i_scores)
        print("[*] EER: {}".format(calc_eer(x, y)))
        EER = "%.3f%%" % (calc_eer(x, y) * 100)
        l_eer.append(calc_eer(x, y) * 100)
        lines = plt.plot(x, y, label='ROC')
        plt.setp(lines, 'color', color[i], 'linewidth', 3, 'label', label[i] + "; EER: " + str(EER))

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
    return l_eer


def dynamic(cls, fk_score_path, fp_score_path, dynamic_save_path, label, color):
    for c in cls:
        fk_score_file = os.path.join(fk_score_path, "fk_score_" + c + ".npy")
        fp_score_file = os.path.join(fp_score_path, "fp_score_" + c + ".csv")
        dynamic_save_subject = os.path.join(dynamic_save_path, c)
        if not os.path.exists(dynamic_save_subject):
            os.mkdir(dynamic_save_subject)
        # ---------------------- read fk score and normalize
        fk_score = np.load(fk_score_file, allow_pickle=True)[()]
        fk_matrix = np.array(fk_score['mmat'])
        g_scores = np.array(fk_score["g_scores"])
        g_min = np.min(g_scores)
        g_max = np.max(g_scores)
        i_scores = np.array(fk_score['i_scores'])
        i_min = np.min(i_scores)
        i_max = np.max(i_scores)
        # normalize score
        max = np.max(np.array([g_max, i_max]))
        min = np.min(np.array([g_min, i_min]))
        fk_matrix = (fk_matrix - min) / (max - min)
        fk_g_scores, fk_i_scores = get_score(fk_matrix)

        # ---------------------- read fp score and normalize
        fp_score = pd.read_csv(fp_score_file, header=None)
        fp_score = fp_score.drop(columns=0).drop([0])
        fp_matrix = fp_score.values.tolist()
        fp_matrix = np.array(fp_matrix).astype(np.float)
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
        fp_matrix = (fp_matrix - min) / (max - min)
        fp_g_scores, fp_i_scores = get_score(fp_matrix)
        fp_g_valid, fp_i_valid = [], []
        for i in range(len(fp_g_scores)):
            if fp_g_scores[i] < 0:
                fp_g_valid.append(1)
            else:
                fp_g_valid.append(fp_g_scores[i])
        for i in range(len(fp_i_scores)):
            if fp_i_scores[i] < 0:
                fp_i_valid.append(1)
            else:
                fp_i_valid.append(fp_i_scores[i])
        fp_g_scores, fp_i_scores = np.array(fp_g_valid), np.array(fp_i_valid)

        eer = []
        for w in range(0, 105, 5):
            w1 = round(w / 100, 2)
            w2 = round((100 - w) / 100, 2)
            dynamic_g = w1 * fk_g_scores + w2 * fp_g_scores
            dynamic_i = w1 * fk_i_scores + w2 * fp_i_scores
            dynamic_save_file = os.path.join(dynamic_save_subject, str(w1) + '-' + str(w2) + '.pdf')
            l_eer = draw_roc([[fk_g_scores, fk_i_scores], [fp_g_scores, fp_i_scores], [dynamic_g, dynamic_i]],
                             save_path=dynamic_save_file, label=label, color=color)
            eer.append(l_eer[2])

        eer = np.array(eer).astype(np.float)
        min_eer = np.min(eer)
        min_index = np.where(eer == min_eer)
        with open(dynamic_save_path+ '/log.txt', 'a+') as f:
            f.write("For the cls " + str(c) + ": min err value: " + str(min_eer) + " min eer value index: " + str(
            min_index) + '\n')
        print("For the cls " + str(c) + ": min err value: " + str(min_eer) + " min eer value index: " + str(
            min_index) + '\n')


def holistic(cls, fk_score_path, fp_score_path, holistic_save_path, label, color):
    for c in cls:
        fk_score_file = os.path.join(fk_score_path, "fk_score_" + c + ".npy")
        fp_score_file = os.path.join(fp_score_path, "fp_score_" + c + ".csv")
        holistic_save_subject = os.path.join(holistic_save_path, c)
        if not os.path.exists(holistic_save_subject):
            os.mkdir(holistic_save_subject)
        # ---------------------- read fk score and normalize
        fk_score = np.load(fk_score_file, allow_pickle=True)[()]
        fk_matrix = np.array(fk_score['mmat'])
        g_scores = np.array(fk_score["g_scores"])
        g_min = np.min(g_scores)
        g_max = np.max(g_scores)
        i_scores = np.array(fk_score['i_scores'])
        i_min = np.min(i_scores)
        i_max = np.max(i_scores)
        # normalize score
        max = np.max(np.array([g_max, i_max]))
        min = np.min(np.array([g_min, i_min]))
        fk_matrix = (fk_matrix - min) / (max - min)

        fk_g_scores, fk_i_scores = get_score(fk_matrix)

        # ---------------------- read fp score and normalize
        fp_score = pd.read_csv(fp_score_file, header=None)
        fp_score = fp_score.drop(columns=0).drop([0])
        fp_matrix = fp_score.values.tolist()
        fp_matrix = np.array(fp_matrix).astype(np.float)
        fp_g_scores, fp_i_scores = get_score(fp_matrix)
        fp_g_valid = []
        for i in range(len(fp_g_scores)):
            if fp_g_scores[i] >= 0:
                fp_g_valid.append(fp_g_scores[i])
        fp_i_valid = []
        for i in range(len(fp_i_scores)):
            if fp_i_scores[i] >= 0:
                fp_i_valid.append(fp_i_scores[i])
        fp_g_valid, fp_i_valid = np.array(fp_g_valid), np.array(fp_i_valid)
        g_min = np.min(fp_g_valid)
        g_max = np.max(fp_g_valid)
        i_min = np.min(fp_i_valid)
        i_max = np.max(fp_i_valid)
        # normalize score
        max = np.max(np.array([g_max, i_max]))
        min = np.min(np.array([g_min, i_min]))
        fp_matrix = (fp_matrix - min) / (max - min)
        fp_g_scores, fp_i_scores = get_score(fp_matrix)
        fp_g_valid, fp_i_valid = [], []
        for i in range(len(fp_g_scores)):
            if fp_g_scores[i] < 0:
                fp_g_valid.append(1)
            else:
                fp_g_valid.append(fp_g_scores[i])
        for i in range(len(fp_i_scores)):
            if fp_i_scores[i] < 0:
                fp_i_valid.append(1)
            else:
                fp_i_valid.append(fp_i_scores[i])
        fp_g_scores, fp_i_scores = np.array(fp_g_valid), np.array(fp_i_valid)

        eer = []
        for w in range(0, 105, 5):
            w1 = round(w / 100, 2)
            w2 = round(1-w1, 2)
            holistic_g = (w1 * fk_g_scores + w2 * fp_g_scores) * (1 + 1 / (2 - fp_g_scores))
            holistic_i = (w1 * fk_i_scores + w2 * fp_i_scores) * (1 + 1 / (2 - fp_i_scores))
            holistic_save_file = os.path.join(holistic_save_subject, str(w1) + '-' + str(w2) + '.pdf')
            l_eer = draw_roc([[fk_g_scores, fk_i_scores], [fp_g_scores, fp_i_scores], [holistic_g, holistic_i]],
                             save_path=holistic_save_file, label=label, color=color)
            eer.append(l_eer[2])

        eer = np.array(eer).astype(np.float)
        min_eer = np.min(eer)
        min_index = np.where(eer == min_eer)
        with open(holistic_save_path + '/log.txt', 'a+') as f:
            f.write("For the cls " + str(c) + ": min err value: " + str(min_eer) + " min eer value index: " + str(
            min_index) + '\n')
        print("For the cls " + str(c) + ": min err value: " + str(min_eer) + " min eer value index: " + str(
            min_index) + '\n')


def nonlinear(cls, fk_score_path, fp_score_path, nonlinear_save_path, label, color):
    for c in cls:
        fk_score_file = os.path.join(fk_score_path, "fk_score_" + c + ".npy")
        fp_score_file = os.path.join(fp_score_path, "fp_score_" + c + ".csv")
        nonlinear_save_subject = os.path.join(nonlinear_save_path, c)
        if not os.path.exists(nonlinear_save_subject):
            os.mkdir(nonlinear_save_subject)
        # ---------------------- read fk score and normalize
        fk_score = np.load(fk_score_file, allow_pickle=True)[()]
        fk_matrix = np.array(fk_score['mmat'])
        g_scores = np.array(fk_score["g_scores"])
        g_min = np.min(g_scores)
        g_max = np.max(g_scores)
        i_scores = np.array(fk_score['i_scores'])
        i_min = np.min(i_scores)
        i_max = np.max(i_scores)
        # normalize score
        max = np.max(np.array([g_max, i_max]))
        min = np.min(np.array([g_min, i_min]))
        fk_matrix = (fk_matrix - min) / (max - min)

        fk_g_scores, fk_i_scores = get_score(fk_matrix)

        # ---------------------- read fp score and normalize
        fp_score = pd.read_csv(fp_score_file, header=None)
        fp_score = fp_score.drop(columns=0).drop([0])
        fp_matrix = fp_score.values.tolist()
        fp_matrix = np.array(fp_matrix).astype(np.float)
        fp_g_scores, fp_i_scores = get_score(fp_matrix)
        fp_g_valid = []
        for i in range(len(fp_g_scores)):
            if fp_g_scores[i] >= 0:
                fp_g_valid.append(fp_g_scores[i])
        fp_i_valid = []
        for i in range(len(fp_i_scores)):
            if fp_i_scores[i] >= 0:
                fp_i_valid.append(fp_i_scores[i])
        fp_g_valid, fp_i_valid = np.array(fp_g_valid), np.array(fp_i_valid)
        g_min = np.min(fp_g_valid)
        g_max = np.max(fp_g_valid)
        i_min = np.min(fp_i_valid)
        i_max = np.max(fp_i_valid)
        # normalize score
        max = np.max(np.array([g_max, i_max]))
        min = np.min(np.array([g_min, i_min]))
        fp_matrix = (fp_matrix - min) / (max - min)
        fp_g_scores, fp_i_scores = get_score(fp_matrix)
        fp_g_valid, fp_i_valid = [], []
        for i in range(len(fp_g_scores)):
            if fp_g_scores[i] < 0:
                fp_g_valid.append(1)
            else:
                fp_g_valid.append(fp_g_scores[i])
        for i in range(len(fp_i_scores)):
            if fp_i_scores[i] < 0:
                fp_i_valid.append(1)
            else:
                fp_i_valid.append(fp_i_scores[i])
        fp_g_scores, fp_i_scores = np.array(fp_g_valid), np.array(fp_i_valid)

        eer = []
        for w in range(0, 105, 5):
            w1 = round(w / 100, 2) + 1
            w2 = round(1-w1, 2)
            nonlinear_g = np.power((1 + fk_g_scores) / (1 + fp_g_scores), w1) * np.power(
                (1 + fp_g_scores), 2)
            nonlinear_i = np.power((1 + fk_i_scores) / (1 + fp_i_scores), w1) * np.power(
                (1 + fp_i_scores), 2)
            nonlinear_save_file = os.path.join(nonlinear_save_subject, str(w1) + '-' + str(w2) + '.pdf')
            l_eer = draw_roc([[fk_g_scores, fk_i_scores], [fp_g_scores, fp_i_scores], [nonlinear_g, nonlinear_i]],
                             save_path=nonlinear_save_file, label=label, color=color)
            eer.append(l_eer[2])

        eer = np.array(eer).astype(np.float)
        min_eer = np.min(eer)
        min_index = np.where(eer == min_eer)
        with open(nonlinear_save_path + '/log.txt', 'a+') as f:
            f.write("For the cls " + str(c) + ": min err value: " + str(min_eer) + " min eer value index: " + str(
            min_index) + '\n')
        print("For the cls " + str(c) + ": min err value: " + str(min_eer) + " min eer value index: " + str(
            min_index) + '\n')


if __name__ == "__main__":
    label = ['Finger-Knuckle',
             'Fingerprint',
             'Score-Fusion']

    color = ['#ff0000',
             '#000000',
             "#c0c0c0"]

    cls = ["01", "02", "04", "05", "06", "07", "08", "09", "10"]

    fk_score_path = "/media/zhenyuzhou/Data/Project/Finger-Knuckle-2018/Finger-Knuckle-Assistant-Recognition/checkpoint/fusion_score/fk_score/"
    fp_score_path = "/media/zhenyuzhou/Data/Project/Finger-Knuckle-2018/Finger-Knuckle-Assistant-Recognition/checkpoint/fusion_score/valid_fp_score/"
    dynamic_save_path = "/media/zhenyuzhou/Data/Project/Finger-Knuckle-2018/Finger-Knuckle-Assistant-Recognition/checkpoint/fusion_score/dynamic/"
    if os.path.exists(dynamic_save_path):
        shutil.rmtree(dynamic_save_path)
    os.mkdir(dynamic_save_path)
    holistic_save_path = "/media/zhenyuzhou/Data/Project/Finger-Knuckle-2018/Finger-Knuckle-Assistant-Recognition/checkpoint/fusion_score/holistic/"
    if os.path.exists(holistic_save_path):
        shutil.rmtree(holistic_save_path)
    os.mkdir(holistic_save_path)
    nonlinear_save_path = "/media/zhenyuzhou/Data/Project/Finger-Knuckle-2018/Finger-Knuckle-Assistant-Recognition/checkpoint/fusion_score/nonlinear/"
    if os.path.exists(nonlinear_save_path):
        shutil.rmtree(nonlinear_save_path)
    os.mkdir(nonlinear_save_path)

    dynamic(cls, fk_score_path, fp_score_path, dynamic_save_path, label, color)
    holistic(cls, fk_score_path, fp_score_path, holistic_save_path, label, color)
    nonlinear(cls, fk_score_path, fp_score_path, nonlinear_save_path, label, color)
