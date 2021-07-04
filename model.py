import os
import time
import random

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from pytorch_msssim import ssim


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h0 = F.relu(self.conv1_1(X), inplace=True)
        h1 = F.relu(self.conv1_2(h0), inplace=True)
        h2 = F.max_pool2d(h1, kernel_size=2, stride=2)

        h3 = F.relu(self.conv2_1(h2), inplace=True)
        h4 = F.relu(self.conv2_2(h3), inplace=True)
        h5 = F.max_pool2d(h4, kernel_size=2, stride=2)

        h6 = F.relu(self.conv3_1(h5), inplace=True)
        h7 = F.relu(self.conv3_2(h6), inplace=True)
        h8 = F.relu(self.conv3_3(h7), inplace=True)
        h9 = F.max_pool2d(h8, kernel_size=2, stride=2)
        h10 = F.relu(self.conv4_1(h9), inplace=True)
        h11 = F.relu(self.conv4_2(h10), inplace=True)
        conv4_3 = self.conv4_3(h11)
        result = F.relu(conv4_3, inplace=True)

        return result



def load_vgg16(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    vgg = Vgg16()
    vgg.cuda()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))

    return vgg


def compute_vgg_loss(enhanced_result, input_high):
    instance_norm = nn.InstanceNorm2d(512, affine=False)
    vgg = load_vgg16("./model")
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False
    img_fea = vgg(enhanced_result)
    target_fea = vgg(input_high)

    loss = torch.mean((instance_norm(img_fea) - instance_norm(target_fea)) ** 2)

    return loss



class ResidualModule0(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(ResidualModule0, self).__init__()
        self.Relu = nn.LeakyReLU()

        self.conv0 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x
        out0 = self.Relu(self.conv0(x))
        out1 = self.Relu(self.conv1(out0))
        out2 = self.Relu(self.conv2(out1))
        out3 = self.Relu(self.conv3(out2))
        out4 = self.Relu(self.conv4(out3))
        out = self.Relu(self.conv(residual))

        final_out = torch.cat((out, out4), dim=1)

        return final_out


class ResidualModule1(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(ResidualModule1, self).__init__()
        self.Relu = nn.LeakyReLU()

        self.conv0 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x
        out0 = self.Relu(self.conv0(x))
        out1 = self.Relu(self.conv1(out0))
        out2 = self.Relu(self.conv2(out1))
        out3 = self.Relu(self.conv3(out2))
        out4 = self.Relu(self.conv4(out3))
        out = self.Relu(self.conv(residual))

        final_out = torch.cat((out, out4), dim=1)

        return final_out


class ResidualModule2(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(ResidualModule2, self).__init__()
        self.Relu = nn.LeakyReLU()

        self.conv0 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x
        out0 = self.Relu(self.conv0(x))
        out1 = self.Relu(self.conv1(out0))
        out2 = self.Relu(self.conv2(out1))
        out3 = self.Relu(self.conv3(out2))
        out4 = self.Relu(self.conv4(out3))
        out = self.Relu(self.conv(residual))

        final_out = torch.cat((out, out4), dim=1)

        return final_out


class ResidualModule3(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(ResidualModule3, self).__init__()
        self.Relu = nn.LeakyReLU()

        self.conv0 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x
        out0 = self.Relu(self.conv0(x))
        out1 = self.Relu(self.conv1(out0))
        out2 = self.Relu(self.conv2(out1))
        out3 = self.Relu(self.conv3(out2))
        out4 = self.Relu(self.conv4(out3))
        out = self.Relu(self.conv(residual))

        final_out = torch.cat((out, out4), dim=1)

        return final_out


class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()

        self.activation = nn.LeakyReLU()

        self.conv0 = nn.Conv2d(4, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.RM0 = ResidualModule0()
        self.RM1 = ResidualModule1()
        self.RM2 = ResidualModule2()
        self.RM3 = ResidualModule3()
        self.conv1 = nn.Conv2d(channel * 2, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel * 2, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel * 2, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv5 = nn.Conv2d(channel, 4, kernel_size=1, stride=1, padding=0)

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)

        out0 = self.activation(self.conv0(input_img))
        out1 = self.RM0(out0)
        out2 = self.activation(self.conv1(out1))
        out3 = self.RM1(out2)
        out4 = self.activation(self.conv2(out3))
        out5 = self.RM2(out4)
        out6 = self.activation(self.conv3(out5))
        out7 = self.RM3(out6)
        out8 = self.activation(self.conv4(out7))
        out9 = self.activation(self.conv5(out8))

        R = torch.sigmoid(out9[:, 0:3, :, :])
        L = torch.sigmoid(out9[:, 3:4, :, :])

        return R, L



class DenoiseNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DenoiseNet, self).__init__()
        self.Relu = nn.LeakyReLU()
        self.Denoise_conv0_1 = nn.Conv2d(4, channel, kernel_size, padding=2, padding_mode='replicate', dilation=2)
        self.Denoise_conv0_2 = nn.Conv2d(channel, channel, kernel_size, padding=2, padding_mode='replicate',
                                         dilation=2)  # 96*96
        self.conv0 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Denoise_subsampling0 = nn.Conv2d(channel*2, channel, kernel_size=2, stride=2, padding=0)  # 48*48
        self.conv5 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv7 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv8 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv9 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Denoise_subsampling1 = nn.Conv2d(channel*2, channel, kernel_size=2, stride=2, padding=0)  # 24*24
        self.conv10 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv11 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv12 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv13 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv14 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Denoise_subsampling2 = nn.Conv2d(channel*2, channel, kernel_size=2, stride=2, padding=0)  # 12*12
        self.conv15 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv16 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv17 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv18 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv19 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Denoise_deconv0 = nn.ConvTranspose2d(channel*2, channel, kernel_size=2, stride=2, padding=0,
                                                  output_padding=0)  # 24*24
        self.conv20 = nn.Conv2d(channel * 2, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv21 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv22 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv23 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv24 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Denoise_deconv1 = nn.ConvTranspose2d(channel*2, channel, kernel_size=2, stride=2, padding=0,
                                                  output_padding=0)  # 48*48
        self.conv25 = nn.Conv2d(channel * 2, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv26 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv27 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv28 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv29 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Denoise_deconv2 = nn.ConvTranspose2d(channel*2, channel, kernel_size=2, stride=2, padding=0,
                                                  output_padding=0)  # 96*96

        self.Denoiseout0 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.Denoiseout1 = nn.Conv2d(channel, 3, kernel_size=1, stride=1)


    def forward(self, input_L, input_R):
        input_img = torch.cat((input_R, input_L), dim=1)
        out0 = self.Relu(self.Denoise_conv0_1(input_img))
        out1 = self.Relu(self.Denoise_conv0_2(out0))
        out2 = self.Relu(self.conv0(out1))
        out3 = self.Relu(self.conv1(out2))
        out4 = self.Relu(self.conv2(out3))
        out5 = self.Relu(self.conv3(out4))
        out6 = self.Relu(self.conv4(out5))
        down0 = self.Relu(self.Denoise_subsampling0(torch.cat((out1, out6), dim=1)))
        out7 = self.Relu(self.conv5(down0))
        out8 = self.Relu(self.conv6(out7))
        out9 = self.Relu(self.conv7(out8))
        out10 = self.Relu(self.conv8(out9))
        out11 = self.Relu(self.conv9(out10))
        down1 = self.Relu(self.Denoise_subsampling1(torch.cat((down0, out11), dim=1)))
        out12 = self.Relu(self.conv10(down1))
        out13 = self.Relu(self.conv11(out12))
        out14 = self.Relu(self.conv12(out13))
        out15 = self.Relu(self.conv13(out14))
        out16 = self.Relu(self.conv14(out15))
        down2 = self.Relu(self.Denoise_subsampling2(torch.cat((down1, out16), dim=1)))
        out17 = self.Relu(self.conv15(down2))
        out18 = self.Relu(self.conv16(out17))
        out19 = self.Relu(self.conv17(out18))
        out20 = self.Relu(self.conv18(out19))
        out21 = self.Relu(self.conv19(out20))
        up0 = self.Relu(self.Denoise_deconv0(torch.cat((down2, out21), dim=1)))
        out22 = self.Relu(self.conv20(torch.cat((up0, out16), dim=1)))
        out23 = self.Relu(self.conv21(out22))
        out24 = self.Relu(self.conv22(out23))
        out25 = self.Relu(self.conv23(out24))
        out26 = self.Relu(self.conv24(out25))
        up1 = self.Relu(self.Denoise_deconv1(torch.cat((up0, out26), dim=1)))
        out27 = self.Relu(self.conv25(torch.cat((up1, out11), dim=1)))
        out28 = self.Relu(self.conv26(out27))
        out29 = self.Relu(self.conv27(out28))
        out30 = self.Relu(self.conv28(out29))
        out31 = self.Relu(self.conv29(out30))
        up2 = self.Relu(self.Denoise_deconv2(torch.cat((up1, out31), dim=1)))
        out32 = self.Relu(self.Denoiseout0(torch.cat((out6, up2), dim=1)))
        out33 = self.Relu(self.Denoiseout1(out32))
        denoise_R = out33

        return denoise_R


class EnhanceModule0(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(EnhanceModule0, self).__init__()

        self.Relu = nn.LeakyReLU()
        self.Enhance_conv0_1 = nn.Conv2d(4, channel, kernel_size, padding=2, padding_mode='replicate', dilation=2)
        self.Enhance_conv0_2 = nn.Conv2d(channel, channel, kernel_size, padding=2, padding_mode='replicate',
                                         dilation=2)  # 96*96
        self.conv0 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Enhance_subsampling0 = nn.Conv2d(channel * 2, channel, kernel_size=2, stride=2, padding=0)  # 48*48
        self.conv5 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv7 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv8 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv9 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Enhance_subsampling1 = nn.Conv2d(channel * 2, channel, kernel_size=2, stride=2, padding=0)  # 24*24
        self.conv10 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv11 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv12 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv13 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv14 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Enhance_subsampling2 = nn.Conv2d(channel * 2, channel, kernel_size=2, stride=2, padding=0)  # 12*12
        self.conv15 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv16 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv17 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv18 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv19 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Enhance_deconv0 = nn.ConvTranspose2d(channel * 2, channel, kernel_size=2, stride=2, padding=0,
                                                  output_padding=0)  # 24*24
        self.conv20 = nn.Conv2d(channel * 2, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv21 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv22 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv23 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv24 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Enhance_deconv1 = nn.ConvTranspose2d(channel * 2, channel, kernel_size=2, stride=2, padding=0,
                                                  output_padding=0)  # 48*48
        self.conv25 = nn.Conv2d(channel * 2, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv26 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv27 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv28 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv29 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Enhance_deconv2 = nn.ConvTranspose2d(channel * 2, channel, kernel_size=2, stride=2, padding=0,
                                                  output_padding=0)  # 96*96

        self.Enhanceout0 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.Enhanceout1 = nn.Conv2d(channel * 4, channel, kernel_size, padding=1, padding_mode='replicate')
        self.Enhanceout2 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)


    def forward(self, input_img):
        out0 = self.Relu(self.Enhance_conv0_1(input_img))
        out1 = self.Relu(self.Enhance_conv0_2(out0))
        out2 = self.Relu(self.conv0(out1))
        out3 = self.Relu(self.conv1(out2))
        out4 = self.Relu(self.conv2(out3))
        out5 = self.Relu(self.conv3(out4))
        out6 = self.Relu(self.conv4(out5))
        down0 = self.Relu(self.Enhance_subsampling0(torch.cat((out1, out6), dim=1)))
        out7 = self.Relu(self.conv5(down0))
        out8 = self.Relu(self.conv6(out7))
        out9 = self.Relu(self.conv7(out8))
        out10 = self.Relu(self.conv8(out9))
        out11 = self.Relu(self.conv9(out10))
        down1 = self.Relu(self.Enhance_subsampling1(torch.cat((down0, out11), dim=1)))
        out12 = self.Relu(self.conv10(down1))
        out13 = self.Relu(self.conv11(out12))
        out14 = self.Relu(self.conv12(out13))
        out15 = self.Relu(self.conv13(out14))
        out16 = self.Relu(self.conv14(out15))
        down2 = self.Relu(self.Enhance_subsampling2(torch.cat((down1, out16), dim=1)))
        out17 = self.Relu(self.conv15(down2))
        out18 = self.Relu(self.conv16(out17))
        out19 = self.Relu(self.conv17(out18))
        out20 = self.Relu(self.conv18(out19))
        out21 = self.Relu(self.conv19(out20))
        up0 = self.Relu(self.Enhance_deconv0(torch.cat((down2, out21), dim=1)))
        out22 = self.Relu(self.conv20(torch.cat((up0, out16), dim=1)))
        out23 = self.Relu(self.conv21(out22))
        out24 = self.Relu(self.conv22(out23))
        out25 = self.Relu(self.conv23(out24))
        out26 = self.Relu(self.conv24(out25))
        up1 = self.Relu(self.Enhance_deconv1(torch.cat((up0, out26), dim=1)))
        out27 = self.Relu(self.conv25(torch.cat((up1, out11), dim=1)))
        out28 = self.Relu(self.conv26(out27))
        out29 = self.Relu(self.conv27(out28))
        out30 = self.Relu(self.conv28(out29))
        out31 = self.Relu(self.conv29(out30))
        up2 = self.Relu(self.Enhance_deconv2(torch.cat((up1, out31), dim=1)))
        out32 = self.Relu(self.Enhanceout0(torch.cat((out6, up2), dim=1)))
        up0_1 = F.interpolate(up0, size=(input_img.size()[2], input_img.size()[3]))
        up1_1 = F.interpolate(up1, size=(input_img.size()[2], input_img.size()[3]))
        up2_1 = F.interpolate(up2, size=(input_img.size()[2], input_img.size()[3]))
        out33 = self.Relu(self.Enhanceout1(torch.cat((up0_1, up1_1, up2_1, out32), dim=1)))
        out34 = self.Relu(self.Enhanceout2(out33))
        Enhanced_I = out34

        return Enhanced_I


class EnhanceModule1(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(EnhanceModule1, self).__init__()

        self.Max_pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Relu = nn.LeakyReLU()

        self.Enhance_conv0_1 = nn.Conv2d(4, channel, kernel_size, padding=2, padding_mode='replicate', dilation=2)
        self.Enhance_conv0_2 = nn.Conv2d(channel, channel, kernel_size, padding=2, padding_mode='replicate',
                                         dilation=2)  # 96*96
        self.conv0 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        #  maxpooling(torch.cat(conv0_input 64, conv4_output 64), dim=1))
        self.conv5 = nn.Conv2d(channel * 2, channel * 2, kernel_size=1, stride=1, padding=0)  # input 128
        self.conv6 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv7 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv8 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv9 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        #  maxpooling(torch.cat(conv5_input 128, conv9_output 64), dim=1))
        self.conv10 = nn.Conv2d(channel * 3, channel * 2, kernel_size=1, stride=1, padding=0)  # input 192
        self.conv11 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv12 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv13 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv14 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        #  maxpooling(torch.cat(conv10_input 192 conv14_output 64), dim=1))
        self.conv15_0 = nn.Conv2d(channel * 4, channel * 2, kernel_size=1, stride=1, padding=0)  # input 256
        self.conv16 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv17 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv18 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv19 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Enhance_deconv0 = nn.ConvTranspose2d(channel * 5, channel, kernel_size=2, stride=2, padding=0,
                                                  output_padding=0)  # 24*24
        self.conv20 = nn.Conv2d(channel * 2, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv21 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv22 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv23 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv24 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Enhance_deconv1 = nn.ConvTranspose2d(channel * 2, channel, kernel_size=2, stride=2, padding=0,
                                                  output_padding=0)  # 48*48
        self.conv25 = nn.Conv2d(channel * 2, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv26 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv27 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv28 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv29 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Enhance_deconv2 = nn.ConvTranspose2d(channel * 2, channel, kernel_size=2, stride=2, padding=0,
                                                  output_padding=0)  # 96*96

        self.Enhanceout0 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.Enhanceout1 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)


    def forward(self, input_img):
        out0 = self.Relu(self.Enhance_conv0_1(input_img))
        out1 = self.Relu(self.Enhance_conv0_2(out0))
        out2 = self.Relu(self.conv0(out1))
        out3 = self.Relu(self.conv1(out2))
        out4 = self.Relu(self.conv2(out3))
        out5 = self.Relu(self.conv3(out4))
        out6 = self.Relu(self.conv4(out5))
        down0 = self.Relu(self.Max_pooling(torch.cat((out1, out6), dim=1)))
        out7 = self.Relu(self.conv5(down0))
        out8 = self.Relu(self.conv6(out7))
        out9 = self.Relu(self.conv7(out8))
        out10 = self.Relu(self.conv8(out9))
        out11 = self.Relu(self.conv9(out10))
        down1 = self.Relu(self.Max_pooling(torch.cat((down0, out11), dim=1)))
        out12 = self.Relu(self.conv10(down1))
        out13 = self.Relu(self.conv11(out12))
        out14 = self.Relu(self.conv12(out13))
        out15 = self.Relu(self.conv13(out14))
        out16 = self.Relu(self.conv14(out15))
        down2 = self.Relu(self.Max_pooling(torch.cat((down1, out16), dim=1)))
        out17 = self.Relu(self.conv15_0(down2))
        out18 = self.Relu(self.conv16(out17))
        out19 = self.Relu(self.conv17(out18))
        out20 = self.Relu(self.conv18(out19))
        out21 = self.Relu(self.conv19(out20))
        up0 = self.Relu(self.Enhance_deconv0(torch.cat((down2, out21), dim=1)))
        out22 = self.Relu(self.conv20(torch.cat((up0, out16), dim=1)))
        out23 = self.Relu(self.conv21(out22))
        out24 = self.Relu(self.conv22(out23))
        out25 = self.Relu(self.conv23(out24))
        out26 = self.Relu(self.conv24(out25))
        up1 = self.Relu(self.Enhance_deconv1(torch.cat((up0, out26), dim=1)))
        out27 = self.Relu(self.conv25(torch.cat((up1, out11), dim=1)))
        out28 = self.Relu(self.conv26(out27))
        out29 = self.Relu(self.conv27(out28))
        out30 = self.Relu(self.conv28(out29))
        out31 = self.Relu(self.conv29(out30))
        up2 = self.Relu(self.Enhance_deconv2(torch.cat((up1, out31), dim=1)))
        out32 = self.Relu(self.Enhanceout0(torch.cat((out6, up2), dim=1)))
        out33 = self.Relu(self.Enhanceout1(out32))
        Enhanced_I = out33

        return Enhanced_I


class EnhanceModule2(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(EnhanceModule2, self).__init__()

        self.Relu = nn.LeakyReLU()
        self.Enhance_conv0_1 = nn.Conv2d(4, channel, kernel_size, padding=2, padding_mode='replicate', dilation=2)
        self.Enhance_conv0_2 = nn.Conv2d(channel, channel, kernel_size, padding=2, padding_mode='replicate',
                                         dilation=2)  # 96*96
        self.conv0 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Enhance_subsampling0 = nn.Conv2d(channel, channel, kernel_size=2, stride=2, padding=0)  # 48*48
        self.conv5 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv7 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv8 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv9 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Enhance_subsampling1 = nn.Conv2d(channel, channel, kernel_size=2, stride=2, padding=0)  # 24*24
        self.conv10 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv11 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv12 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv13 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv14 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Enhance_subsampling2 = nn.Conv2d(channel, channel, kernel_size=2, stride=2, padding=0)  # 12*12
        self.conv15 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv16 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv17 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv18 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv19 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Enhance_deconv0 = nn.ConvTranspose2d(channel, channel, kernel_size=2, stride=2, padding=0,
                                                  output_padding=0)  # 24*24
        self.conv20 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv21 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv22 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv23 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv24 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Enhance_deconv1 = nn.ConvTranspose2d(channel, channel, kernel_size=2, stride=2, padding=0,
                                                  output_padding=0)  # 48*48
        self.conv25 = nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0)
        self.conv26 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv27 = nn.Conv2d(channel * 4, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv28 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv29 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.Enhance_deconv2 = nn.ConvTranspose2d(channel, channel, kernel_size=2, stride=2, padding=0,
                                                  output_padding=0)  # 96*96

        self.Enhanceout0 = nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate')
        self.Enhanceout1 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)


    def forward(self, input_img):
        out0 = self.Relu(self.Enhance_conv0_1(input_img))
        out1 = self.Relu(self.Enhance_conv0_2(out0))
        out2 = self.Relu(self.conv0(out1))
        out3 = self.Relu(self.conv1(out2))
        out4 = self.Relu(self.conv2(out3))
        out5 = self.Relu(self.conv3(out4))
        out6 = self.Relu(self.conv4(out5))
        down0 = self.Relu(self.Enhance_subsampling0(out6))
        out7 = self.Relu(self.conv5(down0))
        out8 = self.Relu(self.conv6(out7))
        out9 = self.Relu(self.conv7(out8))
        out10 = self.Relu(self.conv8(out9))
        out11 = self.Relu(self.conv9(out10))
        down1 = self.Relu(self.Enhance_subsampling1(out11))
        out12 = self.Relu(self.conv10(down1))
        out13 = self.Relu(self.conv11(out12))
        out14 = self.Relu(self.conv12(out13))
        out15 = self.Relu(self.conv13(out14))
        out16 = self.Relu(self.conv14(out15))
        down2 = self.Relu(self.Enhance_subsampling2(out16))
        out17 = self.Relu(self.conv15(down2))
        out18 = self.Relu(self.conv16(out17))
        out19 = self.Relu(self.conv17(out18))
        out20 = self.Relu(self.conv18(out19))
        out21 = self.Relu(self.conv19(out20))
        up0 = self.Relu(self.Enhance_deconv0(out21))
        out22 = self.Relu(self.conv20(up0))
        out23 = self.Relu(self.conv21(out22))
        out24 = self.Relu(self.conv22(out23))
        out25 = self.Relu(self.conv23(out24))
        out26 = self.Relu(self.conv24(out25))
        up1 = self.Relu(self.Enhance_deconv1(out26))
        out27 = self.Relu(self.conv25(up1))
        out28 = self.Relu(self.conv26(out27))
        out29 = self.Relu(self.conv27(out28))
        out30 = self.Relu(self.conv28(out29))
        out31 = self.Relu(self.conv29(out30))
        up2 = self.Relu(self.Enhance_deconv2(out31))
        out32 = self.Relu(self.Enhanceout0(up2))
        out33 = self.Relu(self.Enhanceout1(out32))
        Enhanced_I = out33

        return Enhanced_I


class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()
        self.Relu = nn.LeakyReLU()
        self.conv0 = nn.Conv2d(channel*3, channel, kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.conv1 = nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0)
        self.EnhanceModule0 = EnhanceModule0()
        self.EnhanceModule1 = EnhanceModule1()
        self.EnhanceModule2 = EnhanceModule2()

    def forward(self, input_L, denoise_R):
        input_img = torch.cat((input_L, denoise_R), dim=1)

        out0 = self.EnhanceModule0(input_img)
        out1 = self.EnhanceModule1(input_img)
        out2 = self.EnhanceModule2(input_img)
        out3 = self.Relu(self.conv0(torch.cat((out0, out1, out2), dim=1)))

        output = self.Relu(self.conv1(out3))
        return output


class R2RNet(nn.Module):
    def __init__(self):
        super(R2RNet, self).__init__()

        self.DecomNet = DecomNet()
        self.DenoiseNet = DenoiseNet()
        self.RelightNet = RelightNet()
        self.vgg = load_vgg16("./model")

    def forward(self, input_low, input_high):


        input_low = Variable(torch.FloatTensor(torch.from_numpy(input_low))).cuda()
        input_high = Variable(torch.FloatTensor(torch.from_numpy(input_high))).cuda()
        # Forward DecomNet
        R_low, I_low = self.DecomNet(input_low)
        R_high, I_high = self.DecomNet(input_high)
        # Forward DenoiseNet
        denoise_R = self.DenoiseNet(I_low, R_low)
        # Forward RelightNet
        I_delta = self.RelightNet(I_low, denoise_R)

        # Other variables
        I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)
        I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)
        I_delta_3 = torch.cat((I_delta, I_delta, I_delta), dim=1)

        # DecomNet_loss
        self.vgg_loss = compute_vgg_loss(R_low * I_low_3,  input_low).cuda() + compute_vgg_loss(R_high * I_high_3, input_high).cuda()
        self.recon_loss_low = F.l1_loss(R_low * I_low_3, input_low).cuda()
        self.recon_loss_high = F.l1_loss(R_high * I_high_3, input_high).cuda()
        self.recon_loss_mutal_low = F.l1_loss(R_high * I_low_3, input_low).cuda()
        self.recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, input_high).cuda()

        self.loss_Decom = self.recon_loss_low + \
                          self.recon_loss_high + \
                          self.vgg_loss
        # DenoiseNet_loss
        self.denoise_loss = F.l1_loss(denoise_R, R_high).cuda()
        self.denoise_vgg = compute_vgg_loss(denoise_R, R_high).cuda()
        self.denoise_SSIM = 1 - ssim(denoise_R, R_high, win_size=3).cuda()
        self.loss_Denoise = self.denoise_loss + \
                            self.denoise_SSIM + self.denoise_vgg

        # RelightNet_loss
        self.SSIM_loss = 1 - ssim(denoise_R * I_delta_3, input_high, win_size=3).cuda()
        self.Relight_loss = F.l1_loss(denoise_R * I_delta_3, input_high).cuda()
        self.Relight_vgg = compute_vgg_loss(denoise_R * I_delta_3, input_high).cuda()

        self.loss_Relight = self.Relight_loss + \
                            self.SSIM_loss + self.Relight_vgg

        self.output_R_low = R_low.detach().cpu()
        self.output_I_low = I_low_3.detach().cpu()
        self.output_I_delta = I_delta_3.detach().cpu()
        self.output_R_denoise = denoise_R.detach().cpu()
        self.output_S = denoise_R.detach().cpu() * I_delta_3.detach().cpu()

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def evaluate(self, epoch_num, eval_low_data_names, vis_dir, train_phase):
        print("Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data_names)):
            eval_low_img = Image.open(eval_low_data_names[idx])
            eval_low_img = np.array(eval_low_img, dtype="float32")/255.0
            eval_low_img = np.transpose(eval_low_img, (2, 0, 1))
            input_low_eval = np.expand_dims(eval_low_img, axis=0)

            if train_phase == "Decom":
                self.forward(input_low_eval, input_low_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                input = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                cat_image = np.concatenate([input, result_1, result_2], axis=2)
            if train_phase == 'Denoise':
                self.forward(input_low_eval, input_low_eval)
                result_1 = self.output_R_denoise
                input = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                cat_image = np.concatenate([input, result_1], axis=2)
            if train_phase == "Relight":
                self.forward(input_low_eval, input_low_eval)
                result_1 = self.output_R_denoise
                result_2 = self.output_I_low
                result_3 = self.output_I_delta
                result_4 = self.output_S
                input = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                result_3 = np.squeeze(result_3)
                result_4 = np.squeeze(result_4)
                cat_image = np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)

            cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
            im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
            filepath = os.path.join(vis_dir, 'eval_%s_%d_%d.png' %
                       (train_phase, idx + 1, epoch_num))
            im.save(filepath[:-4] + '.jpg')


    def save(self, iter_num, ckpt_dir):
        save_dir = ckpt_dir + '/' + self.train_phase + '/'
        save_name = save_dir + '/' + str(iter_num) + '.tar'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.train_phase == 'Decom':
            torch.save(self.DecomNet.state_dict(), save_name)
        if self.train_phase == 'Denoise':
            torch.save(self.DenoiseNet.state_dict(), save_name)
        if self.train_phase == 'Relight':
            torch.save(self.RelightNet.state_dict(), save_name)

    def load(self, ckpt_dir):
        load_dir = ckpt_dir + '/' + self.train_phase + '/'
        if os.path.exists(load_dir):
            load_ckpts = os.listdir(load_dir)
            load_ckpts.sort()
            load_ckpts = sorted(load_ckpts, key=len)
            if len(load_ckpts) > 0:
                load_ckpt = load_ckpts[-1]
                global_step = int(load_ckpt[:-4])
                ckpt_dict = torch.load(load_dir + load_ckpt)
                if self.train_phase == 'Decom':
                    self.DecomNet.load_state_dict(ckpt_dict)
                if self.train_phase == 'Denoise':
                    self.DenoiseNet.load_state_dict(ckpt_dict)
                if self.train_phase == 'Relight':
                    self.RelightNet.load_state_dict(ckpt_dict)
                return True, global_step
            else:
                return False, 0
        else:
            return False, 0


    def train(self,
              train_low_data_names,
              train_high_data_names,
              eval_low_data_names,
              batch_size,
              patch_size, epoch,
              lr,
              vis_dir,
              ckpt_dir,
              eval_every_epoch,
              train_phase):
        assert len(train_low_data_names) == len(train_high_data_names)
        numBatch = len(train_low_data_names) // int(batch_size)

        # Create the optimizers
        self.train_op_Decom = optim.Adam(self.DecomNet.parameters(),
                                         lr=lr[0], betas=(0.9, 0.999), weight_decay=0.0001)
        self.train_op_Denoise = optim.Adam(self.DenoiseNet.parameters(),
                                           lr=lr[0], betas=(0.9, 0.999), weight_decay=0.0001)
        self.train_op_Relight = optim.Adam(self.RelightNet.parameters(),
                                           lr=lr[0], betas=(0.9, 0.999), weight_decay=0.0001)

        # Initialize a network if its checkpoint is available
        self.train_phase = train_phase
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("No pretrained model to restore!")

        print("Start training for phase %s, with start epoch %d start iter %d : " %
             (self.train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0
        for epoch in range(start_epoch, epoch):
            self.lr = lr[epoch]
            # Adjust learning rate
            for param_group in self.train_op_Decom.param_groups:
                param_group['lr'] = self.lr
            for param_group in self.train_op_Denoise.param_groups:
                param_group['lr'] = self.lr
            for param_group in self.train_op_Relight.param_groups:
                param_group['lr'] = self.lr
            for batch_id in range(start_step, numBatch):
                # Generate training data for a batch
                batch_input_low = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                batch_input_high = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                for patch_id in range(batch_size):
                    # Load images
                    train_low_img = Image.open(train_low_data_names[image_id])
                    train_low_img = np.array(train_low_img, dtype='float32')/255.0
                    train_high_img = Image.open(train_high_data_names[image_id])
                    train_high_img = np.array(train_high_img, dtype='float32')/255.0
                    # Take random crops
                    h, w, _ = train_low_img.shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
                    train_low_img = train_low_img[x: x + patch_size, y: y + patch_size, :]
                    train_high_img = train_high_img[x: x + patch_size, y: y + patch_size, :]
                    # Data augmentation
                    if random.random() < 0.5:
                        train_low_img = np.flipud(train_low_img)
                        train_high_img = np.flipud(train_high_img)
                    if random.random() < 0.5:
                        train_low_img = np.fliplr(train_low_img)
                        train_high_img = np.fliplr(train_high_img)
                    rot_type = random.randint(1, 4)
                    if random.random() < 0.5:
                        train_low_img = np.rot90(train_low_img, rot_type)
                        train_high_img = np.rot90(train_high_img, rot_type)
                    # Permute the images to tensor format
                    train_low_img = np.transpose(train_low_img, (2, 0, 1))
                    train_high_img = np.transpose(train_high_img, (2, 0, 1))
                    # Prepare the batch
                    batch_input_low[patch_id, :, :, :] = train_low_img
                    batch_input_high[patch_id, :, :, :] = train_high_img
                    self.input_low = batch_input_low
                    self.input_high = batch_input_high

                    image_id = (image_id + 1) % len(train_low_data_names)
                    if image_id == 0:
                        tmp = list(zip(train_low_data_names, train_high_data_names))
                        random.shuffle(list(tmp))
                        train_low_data_names, train_high_data_names = zip(*tmp)


                # Feed-Forward to the network and obtain loss
                self.forward(self.input_low,  self.input_high)
                if self.train_phase == "Decom":
                    self.train_op_Decom.zero_grad()
                    self.loss_Decom.backward()
                    self.train_op_Decom.step()
                    loss = self.loss_Decom.item()
                if self.train_phase == 'Denoise':
                    self.train_op_Denoise.zero_grad()
                    self.loss_Denoise.backward()
                    self.train_op_Denoise.step()
                    loss = self.loss_Denoise.item()
                elif self.train_phase == "Relight":
                    self.train_op_Relight.zero_grad()
                    self.loss_Relight.backward()
                    self.train_op_Relight.step()
                    loss = self.loss_Relight.item()

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            # Evaluate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data_names,
                              vis_dir=vis_dir, train_phase=train_phase)
                self.save(iter_num, ckpt_dir)

        print("Finished training for phase %s." % train_phase)


    def predict(self,
                test_low_data_names,
                res_dir,
                ckpt_dir):

        # Load the network with a pre-trained checkpoint
        self.train_phase = 'Decom'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, "  : Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        self.train_phase = 'Denoise'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, ": Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        self.train_phase = 'Relight'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, ": Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        # Set this switch to True to also save the reflectance and shading maps
        save_R_L = False

        # Predict for the test images
        for idx in range(len(test_low_data_names)):
            test_img_path = test_low_data_names[idx]
            test_img_name = test_img_path.split('/')[-1]
            print('Processing ', test_img_name)
            test_low_img = Image.open(test_img_path)
            test_low_img = np.array(test_low_img, dtype="float32")/255.0
            test_low_img = np.transpose(test_low_img, (2, 0, 1))
            input_low_test = np.expand_dims(test_low_img, axis=0)

            self.forward(input_low_test, input_low_test)
            result_1 = self.output_R_denoise
            result_2 = self.output_I_low
            result_3 = self.output_I_delta
            result_4 = self.output_S
            input = np.squeeze(input_low_test)
            result_1 = np.squeeze(result_1)
            result_2 = np.squeeze(result_2)
            result_3 = np.squeeze(result_3)
            result_4 = np.squeeze(result_4)
            if save_R_L:
                cat_image = np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)
            else:
                cat_image = result_4.numpy()

            cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
            im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
            filepath = res_dir + '/' + test_img_name
            im.save(filepath[:-4] + '.jpg')
