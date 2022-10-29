import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
from PIL import Image
import numpy as np
import math
import torch
import argparse
from torch.autograd import Variable
import models.loss_function, models.net_model
import models.EfficientNetV2
from protocols.confusionmatrix.protocol_util import *
from torchvision import transforms
from inspect import getsourcefile
from models.EfficientNetV2 import ConvBNAct
from collections import OrderedDict
from functools import partial
import datetime

current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

transform = transforms.Compose([transforms.ToTensor()])

def calc_feats_more(*paths):
    """
    1.Read a batch of images from the given paths
    2.Normalize image from 0-255 to 0-1
    3.Get a batch of feature from the model inference()
    """
    size = args.default_size
    w, h = size[0], size[1]
    container = np.zeros((len(paths), 3, h, w))
    for i, path in enumerate(paths):
        im = np.array(
            Image.open(path).convert('RGB').resize(size=size),
            dtype=np.float32
        )
        # change hxwxc = cxhxw
        im = np.transpose(im, (2, 0, 1))
        container[i, :, :, :] = im
    container /= 255.
    container = torch.from_numpy(container.astype(np.float32))
    container = container.cuda()
    container = Variable(container, requires_grad=False)
    fv = inference(container)

    return fv.cpu().data.numpy()


def genuine_imposter(test_path):
    subs = subfolders(test_path, preserve_prefix=True)
    nsubs = len(subs)
    feats_all = []
    subimnames = []

    starttime = datetime.datetime.now()
    for i, usr in enumerate(subs):
        subims = subimages(usr, preserve_prefix=True)
        subimnames += subims
        nims = len(subims)
        feats_all.append(calc_feats_more(*subims))
    feats_all = torch.from_numpy(np.concatenate(feats_all, 0)).cuda()
    endtime = datetime.datetime.now()
    print("inference time: " + str((endtime-starttime).microseconds))

    inference_time = (endtime-starttime).microseconds

    # nsubs-> how many subjects on the test_path
    # nims-> how many images on each of subjects' path
    # for example, for hd(1-4), matching_matrix.shape = (714x4, 714x4)
    starttime = datetime.datetime.now()
    matching_matrix = np.ones((nsubs * nims, nsubs * nims)) * 1000000
    for i in range(1, feats_all.size(0)):
        feat1 = feats_all[:-i, :, :, :]
        feat2 = feats_all[i:, :, :, :]
        loss = _loss(feat1, feat2)
        matching_matrix[:-i, i] = loss
    endtime = datetime.datetime.now()
    print("matching time: " + str((endtime-starttime).microseconds))
    matching_time = (endtime-starttime).microseconds

    return inference_time, matching_time


parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str,
                    default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/Finger-Knuckle-Assistant-Recognition/dataset/test/",
                    dest="test_path")
parser.add_argument("--model_path", type=str,
                    default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/Finger-Knuckle-Assistant-Recognition/checkpoint/EfficientNetV2-S/fkv3(yolov5)-105-221_EfficientNetV2-S-wholeimagerotationandtranslation-lr0.001-subs8-angle5-a20-s4_2022-06-15-17-40/ckpt_epoch_1340.pth",
                    dest="model_path")
parser.add_argument("--default_size", type=int, dest="default_size", default=(300, 300))
parser.add_argument("--shift_size", type=int, dest="shift_size", default=4)
parser.add_argument('--block_size', type=int, dest="block_size", default=8)
parser.add_argument("--rotate_angle", type=int, dest="rotate_angle", default=4)
parser.add_argument("--top_k", type=int, dest="top_k", default=16)
parser.add_argument("--save_mmat", type=bool, dest="save_mmat", default=True)
parser.add_argument('--model', type=str, dest='model', default="EfficientNetV2-S")

model_dict = {
    "RFNet": models.net_model.ResidualFeatureNet().cuda(),
    "DeConvRFNet": models.net_model.DeConvRFNet().cuda(),
    "EfficientNetV2-S": models.EfficientNetV2.efficientnetv2_s().cuda(),
    "RFNWithSTNet": models.net_model.RFNWithSTNet().cuda(),
    "ConvNet": models.net_model.ConvNet().cuda(),
}

args = parser.parse_args()
inference = model_dict[args.model].cuda()
if args.model == "EfficientNetV2-S":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_trained = "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/" \
                  "Finger-Knuckle-Assistant-Recognition/checkpoint/EfficientNetV2-S/pre_efficientnetv2-s.pth"
    weights_dict = torch.load(pre_trained, map_location=device)
    load_weights_dict = {k: v for k, v in weights_dict.items()
                         if inference.state_dict()[k].numel() == v.numel()}
    print(inference.load_state_dict(load_weights_dict, strict=False))
    norm_layer = partial(torch.nn.BatchNorm2d, eps=1e-3, momentum=0.1)
    fk_head = OrderedDict()
    # head_input_c of efficientnet_s: 512
    fk_head.update({"conv1": ConvBNAct(256,
                                       64,
                                       kernel_size=3,
                                       norm_layer=norm_layer)})  # 激活函数默认是SiLU

    fk_head.update({"conv2": ConvBNAct(64,
                                       1,
                                       kernel_size=3,
                                       norm_layer=norm_layer)})

    fk_head = torch.nn.Sequential(fk_head)
    inference.head = fk_head
    inference = inference.cuda().eval()

inference.load_state_dict(torch.load(args.model_path))
# inference = torch.jit.load("knuckle-script-polyu.pt")
Loss = models.loss_function.ShiftedLoss(args.shift_size, args.shift_size)
# Loss = models.loss_function.WholeImageRotationAndTranslation(args.shift_size, args.shift_size, args.rotate_angle)
# Loss = models.loss_function.ImageBlockRotationAndTranslation(i_block_size=args.block_size, i_v_shift=args.shift_size,
#                                                              i_h_shift=args.shift_size, i_angle=args.rotate_angle,
#                                                              i_topk=args.top_k)
Loss.cuda()
Loss.eval()


def _loss(feats1, feats2):
    loss = Loss(feats1, feats2)
    if isinstance(loss, torch.autograd.Variable):
        loss = loss.data
    return loss.cpu().numpy()


inference = inference.cuda()
inference.eval()
total_inference = 0
total_matching = 0
for i in range(10):
    inference_time, matching_time = genuine_imposter(args.test_path)
    total_inference += inference_time
    total_matching += matching_time

print("average inference time: " + str(total_inference/10))
print("average matching time: " + str(total_matching/10))