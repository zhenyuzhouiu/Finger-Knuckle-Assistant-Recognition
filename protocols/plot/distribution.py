# =========================================================
# @ Basic Function File: distribution.py
#
# @ calc_coordinates: from genuine and imposter scores to
#   get the distribution of genuine and imposter scores
# =========================================================

import matplotlib.pyplot as plt
import numpy as np

src_npy = "../../output/ConvNetVSRFNet/RFN-WS-protocol3.npy"

data = np.load(src_npy, allow_pickle=True).item()
g_scores = data['g_scores']
i_scores = data['i_scores']
min_scores = min(min(g_scores), min(i_scores))
max_scores = max(max(g_scores), max(i_scores))
fig, ax = plt.subplots(2, 1)
ax[0].hist(g_scores)
ax[0].set_title("G_Score Distribution")
# AxesSubplot object has no attribute 'xlim'
# ax.xlim((min_scores, max_scores))
ax[0].axis(xmin=min_scores, xmax=max_scores)
ax[0].set_xlabel("G_Score")
ax[0].set_ylabel("No. of G_Score")

ax[1].hist(i_scores)
ax[1].set_title("I_Score Distribution")
ax[1].axis(xmin=min_scores, xmax=max_scores)
ax[1].set_xlabel("I_Score")
ax[1].set_ylabel("No. of I_Score")
# forbid title overlap between two subplots
plt.suptitle(src_npy)
plt.tight_layout()



src_npy = "../../output/ConvNetVSRFNet/fkv3-session2_ConvNetEfficientSTNetBinaryConvNet-hammingdistance-a17.npy"

data = np.load(src_npy, allow_pickle=True).item()
g_scores = data['g_scores']
i_scores = data['i_scores']
min_scores = min(min(g_scores), min(i_scores))
max_scores = max(max(g_scores), max(i_scores))
fig2, ax = plt.subplots(2, 1)
ax[0].hist(g_scores)
ax[0].set_title("G_Score Distribution")
ax[0].axis(xmin=min_scores, xmax=max_scores)
ax[0].set_xlabel("G_Score")
ax[0].set_ylabel("No. of G_Score")

ax[1].hist(i_scores)
ax[1].set_title("I_Score Distribution")
ax[1].axis(xmin=min_scores, xmax=max_scores)
ax[1].set_xlabel("I_Score")
ax[1].set_ylabel("No. of I_Score")
plt.suptitle(src_npy)
plt.tight_layout()

plt.show()