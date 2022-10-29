# =========================================================
# @ Basic Function File: plotroc_basic.py
#
# @ calc_coordinates: from genuine and imposter scores to
#   x, y coordinates
# @ calc_eer: from roc x, y coordinates to calculate EER
#
# @ Notes: feel free to change step_size to fast calculate
#   x, y coordinates especisally when imposter numbers are
#   large.
# =========================================================
from __future__ import division
import numpy as np
import math

def calc_coordinates(g_scores, i_scores, step_size=1):
    i_scores = sorted(i_scores)
    isize, gsize = len(i_scores), len(g_scores)
    x, y = [], []
    for i in range(0, isize, step_size):
        x.append(math.log10((i + 1) / isize))
        y.append(len(g_scores[g_scores < i_scores[i]]) / gsize)
    return x, y

def calc_eer(x, y):
    for xi, yi in zip(x, y):
        if 1 - math.pow(10, xi) < yi:
            return math.pow(10, xi)


