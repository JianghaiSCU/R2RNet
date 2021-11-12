# R2RNet
Official code of "R2RNet: Low-light Image Enhancement via Real-low to Real-normal Network." Jiang Hai, Zhu Xuan, Ren Yang, Yutong Hao, Fengzhu Zou, Fang Lin, and Songchen Han(Submitted to IEEE transaction on Image Processing)

Paper link: https://arxiv.org/abs/2106.14501
## Network Architecture
![fig3](https://user-images.githubusercontent.com/86350392/141397717-abff2d12-e810-4744-96e2-a1ce6af87002.jpeg)
The proposed R2RNet architecture. Our network consists of three subnets: a Decom-Net, a Denoise-Net, and a Enhance-Net, which perform decomposing, denoising, contrast enhancement and detail preservation, respectively. The Decom-Net decomposes the low-light image into an illumination map and a reflectance map based on the Retinex theory. The Denoise-Net aims to suppress the noise in the reflectance map. Subsequently, the illumination map obtained by Decom-Net and the reflectance map obtained by Denoise-Net will be sent to the Relight-Net to improve image contrast and reconstruct details.
![fig4](https://user-images.githubusercontent.com/86350392/141397881-334d4764-5fe0-4412-9e87-fef882089c53.jpeg)
The proposed Relight-Net architecture. The Relight-Net consists of two modules: Contrast Enhancement Module (CEM) and Detail Reconstruction Module (DRM). CEM uses spatial information for contrast enhancement and DRM uses frequency information to preserve image details.

# Pytorch
This is a Pytorch implementation of R2RNet.
## Requirements
1. Python 3.x 
2. Pytorch 1.x.0
## Dataset
You can download the LSRW dataset from: https://pan.baidu.com/s/1UxFllrtRSh4E8ir8LdTb9w (code: wmr1) 
If you use our code and  dataset, please cite our paper.
## Pre-trained model
We have modified the code to obtain better performance. We will release the updated code as soon as possible.
## Testing Usage
python predict.py
## Training Usage
python trian.py
# Reference
Code borrows heavily from https://github.com/aasharma90/RetinexNet_PyTorch.
