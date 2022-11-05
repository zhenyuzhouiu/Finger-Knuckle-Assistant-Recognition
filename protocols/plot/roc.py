# =========================================================
# @ Plot File: IITD under IITD-like protocol
# =========================================================

from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

from plotroc_basic import *


src_npy = ['/media/zhenyuzhou/Data/Project/Finger-Knuckle-2018/Finger-Knuckle-Assistant-Recognition/checkpoint/Joint-Finger-RFNet/Joint-Left-Middle_AssistantModel-wholeimagerotationandtranslation-lr1e-05-subs8-angle0-a20-hs0_vs0_2022-11-04-22-46/output/leave-one-out/index-protocol.npy',
           '/media/zhenyuzhou/Data/Project/Finger-Knuckle-2018/Finger-Knuckle-Assistant-Recognition/checkpoint/Joint-Finger-RFNet/Joint-Left-Middle_AssistantModel-wholeimagerotationandtranslation-lr1e-05-subs8-angle0-a20-hs0_vs0_2022-11-04-22-46/output/leave-one-out/ring-protocol.npy',
           '/media/zhenyuzhou/Data/Project/Finger-Knuckle-2018/Finger-Knuckle-Assistant-Recognition/checkpoint/Joint-Finger-RFNet/Joint-Left-Middle_AssistantModel-wholeimagerotationandtranslation-lr1e-05-subs8-angle0-a20-hs0_vs0_2022-11-04-22-46/output/leave-one-out/little-protocol.npy',
           '/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/left-yolov5s-crop-feature-detection/matching_matrix/left_index_matching_matrix.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/rfn-tl/fkv3-two-session-protocol.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/rfn-tl/hd-protocol.npy',
           '/home/zhenyuzhou/Desktop/Finger-Knuckle-Assistant-Recognition/output/ConvNetVSRFNet/fkv3-session2_Net-shiftedloss-lr0.01-protocol.npy',
           '/home/zhenyuzhou/Desktop/Finger-Knuckle-Assistant-Recognition/output/ConvNetVSRFNet/fkv3-session2_STNetConvNetEfficientNet-shiftedloss-lr0.01-protocol.npy',
           '/home/zhenyuzhou/Desktop/Finger-Knuckle-Assistant-Recognition/output/ConvNetVSRFNet/fkv3-session2_STNetConvNetEfficientNet-shiftedloss-lr1e-06-protocol.npy',
           '/home/zhenyuzhou/Desktop/Finger-Knuckle-Assistant-Recognition/output/ConvNetVSRFNet/fkv3-session2_STNetConvNetEfficientNet-shiftedloss-lr1e-07-protocol.npy',]

label = ['Left-Index',
         'Left-Ring',
         'Left-Little',
         'Left-Index',
         'FirstSTNetThenConvNetMiddleEfficientNet-shiftedloss-lr0.01-500',
         'FirstSTNetThenConvNetMiddleEfficientNet-shiftedloss-lr0.01',
         'Net-shiftedloss-lr0.01',
         'STNetConvNetEfficientNet-shiftedloss-lr0.01',
         'STNetConvNetEfficientNet-shiftedloss-lr1e-06',
         'STNetConvNetEfficientNet-shiftedloss-lr1e-07',
         ]

color = ['#ff0000',
         '#000000',
         "#c0c0c0",
         '#008000',
         '#008080',
         '#000080',
         '#00ffff',
         '#800000',
         '#800080',
         '#808000',
         '#ff00ff',
         '#ff0000']
dst = '/media/zhenyuzhou/Data/Project/Finger-Knuckle-2018/Finger-Knuckle-Assistant-Recognition/checkpoint/Joint-Finger-RFNet/Joint-Left-Middle_AssistantModel-wholeimagerotationandtranslation-lr1e-05-subs8-angle0-a20-hs0_vs0_2022-11-04-22-46/output/leave-one-out/roc.pdf'

for i in range(3):
    data = np.load(src_npy[i], allow_pickle=True)[()]
    g_scores = np.array(data['g_scores'])
    i_scores = np.array(data['i_scores'])

    print ('[*] Source file: {}'.format(src_npy[i]))
    print ('[*] Target output file: {}'.format(dst))
    print ("[*] #Genuine: {}\n[*] #Imposter: {}".format(len(g_scores), len(i_scores)))

    x, y = calc_coordinates(g_scores, i_scores)
    print ("[*] EER: {}".format(calc_eer(x, y)))
    EER = "%.3f%%" % (calc_eer(x, y) * 100)

    # xmin, xmax = plt.xlim()
    # ymin, ymax = plt.ylim()

    lines = plt.plot(x, y, label='ROC')
    plt.setp(lines, 'color', color[i], 'linewidth', 3, 'label', label[i] + "; EER: " + str(EER))

    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)

    plt.grid(True)
    plt.xlabel(r'False Accept Rate', fontsize=18)
    plt.ylabel(r'Genuine Accept Rate', fontsize=18)
    legend = plt.legend(loc='lower right', shadow=False, prop={'size': 16})
    plt.xlim(xmin=min(x))
    plt.xlim(xmax=0)
    plt.ylim(ymax=1)
    plt.ylim(ymin=0.4)

    ax=plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.0)

    plt.xticks(np.array([-4 , -2, 0]), ['$10^{-4}$', '$10^{-2}$', '$10^{0}$'], fontsize=16)
    plt.yticks(np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]), fontsize=16)

if dst:
    plt.savefig(dst, bbox_inches='tight')