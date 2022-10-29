import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import torch
import argparse
import models.net_model
import models.EfficientNetV2
import numpy as np

from torchvision import transforms
from inspect import getsourcefile
from models.EfficientNetV2 import ConvBNAct
from collections import OrderedDict
from functools import partial
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

transform = transforms.Compose([transforms.ToTensor()])

parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str,
                    default="C:\\Users\\ZhenyuZHOU\\Pictures\\50_4-0.jpg",
                    dest="test_path")

parser.add_argument("--model_path", type=str,
                    default="C:\\Users\\ZhenyuZHOU\\Desktop\\Finger-Knuckle-Assistant-Recognition\\checkpoint\\RFNet\\fkv3(yolov5)-105-221_RFNet-wholeimagerotationandtranslation-lr0.001-subs8-angle8-a20-s8_2022-06-27-21-12\\ckpt_epoch_1380.pth",
                    dest="model_path")

parser.add_argument("--default_size", type=int, dest="default_size", default=128)
parser.add_argument('--model', type=str, dest='model', default="RFNet")

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

"""
1.Read a image from given a path
2.Normalize image from 0-255 to 0-1
3.Get the feature map from the model with inference()
"""
container = np.zeros((1, 3, args.default_size, args.default_size))
im = np.array(
    Image.open(args.test_path).convert('RGB').resize((args.default_size, args.default_size)),
    dtype=np.float32
)
# change [h,w,c] = [c,h,w]
im = np.transpose(im, (2, 0, 1))
container[0, :, :, :] = im
container /= 255.
container = torch.from_numpy(container.astype(np.float32))
container = container.cuda()
container = Variable(container, requires_grad=False)
fv = inference(container).squeeze(0)
# fv_vector = fv.reshape(1, -1)
# mean = fv_vector.mean()
# std = fv_vector.std()
# norm_fv = ((fv_vector-mean)/std).reshape(32, 32)

toPIL = transforms.ToPILImage()
pic = toPIL(fv)
pic.save("C:\\Users\\ZhenyuZHOU\\Pictures\\50_4-0-feature.jpg")
