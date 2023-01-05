# ====================================
# Aims: to check the randomness of pytorch
# Authors: ZHOU, Zhenyu
# Time: 05-01-2023
# ====================================

import torch
import numpy as np
import random


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    random_seed(0)
    features = torch.randn(2, 5)
    print("test: {}".format({features}) + '.\n')
