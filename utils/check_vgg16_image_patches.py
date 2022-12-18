import torch
import os
from models.vgg16_texture_keypoint import *
from data.data_factory import load_image
from torchvision import transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker


def draw_grid(image1, image2, grid=(8, 8)):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    h, w, c = image1.shape
    x_interval = w / grid[1]
    loc = plticker.MultipleLocator(base=x_interval)
    ax1.xaxis.set_major_locator(loc)
    y_interval = h / grid[0]
    loc = plticker.MultipleLocator(base=y_interval)
    ax1.yaxis.set_major_locator(loc)
    ax1.grid(which='major', axis='both', linestyle='-')
    ax1.imshow(image1)

    ax2 = fig.add_subplot(122)
    h, w, c = image2.shape
    x_interval = w / grid[1]
    loc = plticker.MultipleLocator(base=x_interval)
    ax2.xaxis.set_major_locator(loc)
    y_interval = h / grid[0]
    loc = plticker.MultipleLocator(base=y_interval)
    ax2.yaxis.set_major_locator(loc)
    ax2.grid(which='major', axis='both', linestyle='-')
    ax2.imshow(image2)

    fig.show()


def show_tensor(tensor1, tensor2, channels=0):
    tensor1 = tensor1.squeeze(0)
    tensor2 = tensor2.squeeze(0)

    show1 = tensor1[channels, :, :]
    show1 = show1.cpu().numpy()
    show2 = tensor2[channels, :, :]
    show2 = show2.cpu().numpy()

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    # imshow() with 2D array, or 3D array with the third dimension being of shape 3 or 4
    ax1.imshow(show1)
    ax2 = fig.add_subplot(122)
    ax2.imshow(show2)
    fig.show()


if __name__ == "__main__":
    # ------------------ load model
    pre_model = "../checkpoint/Joint-Finger-RFNet/MaskLM_VGG16_quadruplet_ssim-r0-a0.5-2a0.3-hs0_vs0_12-17-17-11-50/ckpt_epoch_3000.pth"
    pre_loss = "../checkpoint/Joint-Finger-RFNet/MaskLM_VGG16_quadruplet_ssim-r0-a0.5-2a0.3-hs0_vs0_12-17-17-11-50/ckpt_epoch_lossk_3000.pth"
    model = VGG16().cuda()
    model.load_state_dict(torch.load(pre_model))
    model.eval()
    loss_texture = FeatureCorrelation().cuda()
    loss_texture.load_state_dict(torch.load(pre_loss))
    model.eval()
    # ------------------ load image
    img_path1 = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/mask-seg/03/001/001-03-1.jpg"
    img_a = load_image(img_path1, options="RGB", size=(128, 128))
    img_path2 = "/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/mask-seg/03/010/010-03-1.jpg"
    img_b = load_image(img_path2, options="RGB", size=(128, 128))
    draw_grid(img_a, img_b, grid=(8, 8))

    img_a = transforms.ToTensor()(img_a).unsqueeze(0).cuda()
    img_b = transforms.ToTensor()(img_b).unsqueeze(0).cuda()
    with torch.no_grad():
        a_texture, a_patch = model(img_a)
        b_texture, b_patch = model(img_b)
        show_tensor(a_texture, b_texture, channels=100)
        show_tensor(a_patch, b_patch, channels=100)
        sinkhorn_d = loss_texture(a_patch, b_patch)
        print("Sinkhorn Distance: " + str(sinkhorn_d) + "\n")
