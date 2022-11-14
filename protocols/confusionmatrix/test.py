import numpy as np

feats_length = [3, 3, 3, 3, 3]
feats_length = np.array(feats_length)
acc_len = np.cumsum(feats_length)
feats_start = acc_len - feats_length
nfeats = acc_len[-1]

g_scores = []
i_scores = []
# nfeats: number of features
for i in range(nfeats):
    # in case of multiple occurrences of the maximum values,
    # the indices corresponding to the first occurrence are returned.
    subj_idx = np.argmax(acc_len > i)
    g_select = [feats_start[subj_idx] + k for k in range(feats_length[subj_idx])]
    for i_i in range(i+1):
        if i_i in g_select:
            g_select.remove(i_i)
    i_select = list(range(nfeats))
    # remove g_select
    for subj_i in range(subj_idx + 1):
        for k in range(feats_length[subj_i]):
            i_select.remove(feats_start[subj_i] + k)