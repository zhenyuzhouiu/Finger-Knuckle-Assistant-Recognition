import os

import argparse
import torch
from PIL import Image
import numpy as np
from torch.autograd import Variable
import scipy.io as io
import math
import datetime


def load_data(data_path, subject_no, set_no):
    init = 0
    for subjectID in range(subject_no):
        for setID in range(set_no):
            path_8 = os.path.join(data_path, str(subjectID+1).zfill(3),
                                str(subjectID+1).zfill(3) + '-2-11-' + str(setID+1) + '-0-8.npy')
            feature_8 = np.load(path_8)
            feature_8 = np.max(feature_8, axis=0)
            feature_8 = np.expand_dims(np.max(feature_8, axis=0), axis=0)


            path_16 = os.path.join(data_path, str(subjectID+1).zfill(3),
                                str(subjectID+1).zfill(3) + '-2-11-' + str(setID+1) + '-0-16.npy')
            feature_16 = np.load(path_16)
            feature_16 = np.max(feature_16, axis=0)
            feature_16 = np.expand_dims(np.max(feature_16, axis=0), axis=0)

            path_32 = os.path.join(data_path, str(subjectID+1).zfill(3),
                                str(subjectID+1).zfill(3) + '-2-11-' + str(setID+1) + '-0-32.npy')
            feature_32 = np.load(path_32)
            feature_32 = np.max(feature_32, axis=0)
            feature_32 = np.expand_dims(np.max(feature_32, axis=0), 0)

            if init == 0:
                data_8 = feature_8
                data_16 = feature_16
                data_32 = feature_32
            else:
                data_8 = np.concatenate([data_8, feature_8], axis=0)
                data_16 = np.concatenate([data_16, feature_16], axis=0)
                data_32 = np.concatenate([data_32, feature_32], axis=0)

            init += 1

    return data_8, data_16, data_32


def class_matrix(data, subject_no, set_no, model, device, save_path):
    score_matrix = np.zeros([subject_no, subject_no*set_no])
    for i in range(subject_no*set_no):
        print(i)
        I = data[i, :, :, :]
        I = I.unsqueeze(0)
        pred = model(I.to(device))
        pred_class = torch.nn.functional.softmax(pred)
        score_matrix[:, i] = pred_class.cpu().numpy()

    score_matrix_path = os.path.join(save_path, "score_matrix.mat")
    io.savemat(score_matrix_path, {'score_matrix':score_matrix})
    score_matrix = -score_matrix

    D_genuine = np.zeros([1, subject_no*set_no])
    D_imposter = np.zeros([1, subject_no*(subject_no-1)*set_no])
    counter_genuine = 0;
    counter_imposter = 0;
    for i in range(score_matrix.shape[0]):
        for j in range(score_matrix.shape[1]):
            subjectID = i
            subjectID_2 = math.floor(j/set_no)
            if(subjectID == subjectID_2):
                D_genuine[:, counter_genuine] = score_matrix[i, j]
                counter_genuine = counter_genuine + 1
            else:
                D_imposter[:, counter_imposter] = score_matrix[i, j]
                counter_imposter = counter_imposter + 1
    D_genuine_path = os.path.join(save_path, "D_genuine.mat")
    io.savemat(D_genuine_path, {'D_genuine':D_genuine})
    D_imposter_path = os.path.join(save_path, "D_imposter.mat")
    io.savemat(D_imposter_path, {'D_imposter': D_imposter})


def feature_vector_single_matrix(data_8, data_16, data_32, subject_no, set_no, save_path):

    matching_matrix = np.ones((subject_no * set_no, subject_no * set_no)) * 1000000
    for i in range(1, subject_no*set_no):
        feat1 = data_8[:-i, :]
        feat2 = data_8[i:, :]
        distance_8 = np.sum((feat1 - feat2)**2, axis=1) / feat1.shape[1]

        feat1 = data_16[:-i, :]
        feat2 = data_16[i:, :]
        distance_16 = np.sum((feat1 - feat2)**2, axis=1) / feat1.shape[1]

        feat1 = data_32[:-i, :]
        feat2 = data_32[i:, :]
        distance_32 = np.sum((feat1 - feat2)**2, axis=1) / feat1.shape[1]

        matching_matrix[:-i, i] = 8*distance_8 + distance_16 + distance_32
        print("[*] Pre-processing matching dict for {} / {} \r".format(i, subject_no*set_no))

    matt = np.ones_like(matching_matrix) * 1000000
    matt[0, :] = matching_matrix[0, :]
    for i in range(1, subject_no*set_no):
        # matching_matrix每行的数值向后移动一位
        matt[i, i:] = matching_matrix[i, :-i]
        for j in range(i):
            # matt[i, j] = matt[j, i]
            matt[i, j] = matching_matrix[j, i - j]

    print("\n [*] Done")

    g_scores = []
    i_scores = []
    for i in range(subject_no * set_no):
        start_idx = int(math.floor(i / set_no))
        start_remainder = int(i % set_no)

        argmin_idx = np.argmin(matt[i, start_idx * set_no: start_idx * set_no + set_no])
        g_scores.append(float(matt[i, start_idx * set_no + argmin_idx]))
        select = list(range(subject_no * set_no))
        # remove genuine matching score
        for j in range(set_no):
            select.remove(start_idx * set_no + j)
        # remove imposter matching scores of same index sample on other subjects
        for j in range(subject_no):
            if j == start_idx:
                continue
            select.remove(j * set_no + start_remainder)
        i_scores += list(np.min(np.reshape(matt[i, select], (-1, set_no - 1)), axis=1))
        print("[*] Processing genuine imposter for {} / {} \r".format(i, subject_no * set_no))
    print("\n [*] Done")
    protocol_path = os.path.join(save_path, 'left_index_matching_matrix.npy')
    np.save(protocol_path, {"g_scores": np.array(g_scores), "i_scores": np.array(i_scores), "mmat": matt})

    return 0


def feature_vector_two_matrix(data, session2, subject_no, set_no, model, device, save_path, num_classes):
    feature_matrix = np.zeros([subject_no * set_no, num_classes])
    feature_matrix_session2 = np.zeros([subject_no * set_no, num_classes])
    for i in range(subject_no * set_no):
        print(i)
        I = data[i, :, :, :]
        I = I.unsqueeze(0)
        pred = model(I.to(device))
        feature_matrix[i, :] = pred.cpu().numpy()
    feature_matrix_path = os.path.join(save_path, "feature_matrix.mat")
    io.savemat(feature_matrix_path, {'feature_matrix': feature_matrix})

    for i in range(subject_no * set_no):
        print(i)
        I = session2[i, :, :, :]
        I = I.unsqueeze(0)
        pred = model(I.to(device))
        feature_matrix_session2[i, :] = pred.cpu().numpy()
    feature_matrix_path = os.path.join(save_path, "feature_matrix_session2.mat")
    io.savemat(feature_matrix_path, {'feature_matrix_session2': feature_matrix_session2})
    feats_gallery = np.concatenate((feature_matrix_session2, feature_matrix_session2), 0)

    nl = subject_no * set_no
    matching_matrix = np.ones((nl, nl)) * 1000000
    for i in range(nl):
        distance = np.sum((feature_matrix - feats_gallery[i:i+nl, :]) ** 2, axis=1) / feature_matrix.shape[1]
        matching_matrix[:, i] = distance
        print("[*] Pre-processing matching dict for {} / {} \r".format(i, nl))

    for i in range(1, nl):
        tmp = matching_matrix[i, -i:].copy()
        matching_matrix[i, i:] = matching_matrix[i, :-i]
        matching_matrix[i, :i] = tmp
    print("\n [*] Done")

    g_scores = []
    i_scores = []
    for i in range(nl):
        start_idx = int(math.floor(i / set_no))
        start_remainder = int(i % set_no)
        g_scores.append(float(np.min(matching_matrix[i, start_idx * set_no: start_idx * set_no + set_no])))
        select = list(range(nl))
        for j in range(set_no):
            select.remove(start_idx * set_no + j)
        i_scores += list(np.min(np.reshape(matching_matrix[i, select], (-1, set_no)), axis=1))
        print("[*] Processing genuine imposter for {} / {} \r".format(i, subject_no * set_no))

    print("\n [*] Done")
    protocol_path = os.path.join(save_path, 'protocol.npy')
    np.save(protocol_path, {"g_scores": np.array(g_scores), "i_scores": np.array(i_scores), "mmat": matching_matrix})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, dest="data_path", default="/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/left-yolov5s-crop-feature-detection/left-index-feature/")
    parser.add_argument("--save_path", type=str, dest="save_path", default="/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/left-yolov5s-crop-feature-detection/matching_matrix/")
    parser.add_argument("--subject_no", type=int, dest="subject_no", default=120)
    parser.add_argument("--set_no", type=int, dest="set_no", default=5)
    args = parser.parse_args()

    data_8, data_16, data_32 = load_data(args.data_path, args.subject_no, args.set_no)
    # session2 = load_data(args.session2, args.subject_no, args.set_no, args.default_size)
    feature_vector_single_matrix(data_8, data_16, data_32,
                                 args.subject_no, args.set_no,
                                 args.save_path)

    # feature_vector_two_matrix(data, session2, args.subject_no, args.set_no, model, device, args.save_path, args.num_classes)
    # class_matrix(data, args.subject_no, args.set_no, model, device, args.save_path)




