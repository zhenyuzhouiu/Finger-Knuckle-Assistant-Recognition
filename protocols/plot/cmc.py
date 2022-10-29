from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse
from functools import reduce
from scipy import io

save_cmc = True

nobject = [6, 6, 4, 6, 6, 4, 6, 6, 4, 4, 6, 6]

src_npy = ['/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/tmp_can_delete/c-rfn-trtl-protocol_940.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/tmp_can_delete/b-rfn-trtl-fkv1_finetune_protocol.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/tmp_can_delete/d1-rfn-trtl-protocol.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/tmp_can_delete/d2-rfn-trtl-protocol.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/tmp_can_delete/e-rfn-trtl-crossthu-protocol.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/rfn-tl/hd-protocol.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/rfn-sttl-rsil/2dof3d-rfn-sttl-protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/rfn-sttl-rsil/2dof3d-rfn-trtl-protocol.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/rfn-sttl-rsil/crossthu-rfn-sttl-crossthu-protocol.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/rfn-sttl-rsil/crossthu-rfn-trtl-crossthu-protocol.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/rfn-sttl-rsil/3dof3d-rfn-sttl-protocol.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/rfn-sttl-rsil/3dof3d-rfn-trtl-protocol.npy']

label = ['DeConvRFN-WRS',
         'DeConvRFN-WS',
         'EfficientNet-WRS',
         'EfficientNet-WS',
         'RFN-WS',
         'RFN-WRS',
         'EfficientNet-WRS',
         'RFN-WS',
         'RFN-128-14',
         'RFN-128-16',
         'RFN',
         'xxx']

color = ['#000000',
         '#000080',
         '#008000',
         '#008080',
         "#c0c0c0",
         '#00ffff',
         '#800000',
         '#800080',
         '#808000',
         '#ff00ff',
         '#ff0000',
         '#080080']

dst = '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet/test/rfn-tl/cmc.eps'

for n in range(6):
    data = np.load(src_npy[n], allow_pickle=True)[()]
    match_dict = np.array(data['mmat'])
    nsamples = np.shape(match_dict)[0]

    genuine_idx = np.arange(nsamples).astype(np.float32)
    genuine_idx = np.expand_dims(np.floor(genuine_idx / nobject[n]) * nobject[n], -1)

    min_idx = match_dict.argsort()

    def calc_cmc(rank):
        match_rank = min_idx[:, :rank]
        matching = []
        for j in range(nobject[n]):
            genuine_tmp = np.repeat(genuine_idx + j, rank, 1)
            matching.append(np.sum((match_rank == genuine_tmp).astype(np.int8), 1))
        acc = reduce(lambda x, y: x + y, matching)
        acc = np.clip(acc, 0, 1)
        return np.sum(acc) / np.shape(match_dict)[0]


    x, y = [], []
    for i in range(1, 11):
        x.append(i)
        y.append(calc_cmc(i))

    print ("[*] Accuracy: {}".format(y[0]))

    save_cmc = True
    if save_cmc:
        import scipy.io
        i_src_npy = src_npy[n]
        scipy.io.savemat(i_src_npy[:i_src_npy.find('.npy')] + "_cmc.mat", mdict={'r': x, 'ac': y})

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    lines = plt.plot(x, y, label='')
    plt.setp(lines, 'color', color[n], 'linewidth', 2, 'label', label[n])

    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)

    plt.grid(True)
    plt.xlabel(r'Rank', fontsize=18)
    plt.ylabel(r'Recognition Rate', fontsize=18)
    legend = plt.legend(loc='lower right', shadow=False, prop={'size': 16})
    plt.xlim(xmin=1)
    plt.xlim(xmax=10)
    plt.ylim(ymax=1)
    plt.ylim(ymin=0.96)

    ax=plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.0)

    plt.xticks([2, 4, 6, 8, 10], fontsize=16)
    plt.yticks(np.array([0.965, 0.970, 0.975, 0.98, 0.985, 0.99, 0.995, 1]), fontsize=16)

if dst:
    plt.savefig(dst, bbox_inches='tight')
