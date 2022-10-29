# EigenCam is gradients-free
import os
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import warnings

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
from models.net_model import ResidualFeatureNet, DeConvRFNet
from models.EfficientNetV2 import efficientnetv2_s, ConvBNAct\

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

subject_path = "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/Finger-Knuckle-Assistant-Recognition/dataset/PolyUHD/yolov5/all/700/"
dst_subject_path = "/home/zhenyuzhou/Desktop/IEEE-Transactions-LaTeX2e-templates-and-instructions/Figures/EigenCAM/PolyUHD/MSE/700/"
if not os.path.exists(dst_subject_path):
    os.mkdir(dst_subject_path)
images_file = os.listdir(subject_path)

for i in images_file:
    image_path = os.path.join(subject_path, i)
    dst_image_path = os.path.join(dst_subject_path, i)
    img = np.array(Image.open(image_path).convert('RGB'))
    img = cv2.resize(img, (128, 128))
    rgb_img = img.copy()
    img = np.float32(img) / 255
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)

    model = ResidualFeatureNet()
    model.load_state_dict(torch.load("/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/Finger-Knuckle-Assistant-Recognition/checkpoint/RFNet-TL/hd(yolov5)_RFNet-wholeimagerotationandtranslation-lr0.001-subs8-angle0-a20-hs0_vs0_2022-07-18-10-18/ckpt_epoch_4980.pth"))

    model.eval()
    model.cuda()
    target_layers = [model.conv5]

    cam = EigenCAM(model, target_layers, use_cuda=True)
    grayscale_cam = cam(tensor)

    grayscale_cam = grayscale_cam[0]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    save_img = Image.fromarray(cam_image)
    save_img.save(dst_image_path)

    result_img = Image.fromarray(np.hstack((rgb_img, cam_image)))
    # result_img.show()
