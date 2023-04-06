# R2RNet
Official code of "R2RNet: Low-light Image Enhancement via Real-low to Real-normal Network." Jiang Hai, Zhu Xuan, Ren Yang, Yutong Hao, Fengzhu Zou, Fang Lin, and Songchen Han. [[journal]](https://www.sciencedirect.com/science/article/pii/S1047320322002322)[[arxiv]](https://arxiv.org/abs/2106.14501)

## Citation
If you use this code or dataset from the paper for your research, please cite our paper:
```
@article{hai2023r2rnet,
  title={R2rnet: Low-light image enhancement via real-low to real-normal network},
  author={Hai, Jiang and Xuan, Zhu and Yang, Ren and Hao, Yutong and Zou, Fengzhu and Lin, Fang and Han, Songchen},
  journal={Journal of Visual Communication and Image Representation},
  volume={90},
  pages={103712},
  year={2023}
}
```

## Network Architecture
![fig3](https://user-images.githubusercontent.com/86350392/141397717-abff2d12-e810-4744-96e2-a1ce6af87002.jpeg)
The proposed R2RNet architecture. Our network consists of three subnets: a Decom-Net, a Denoise-Net, and a Enhance-Net, which perform decomposing, denoising, contrast enhancement and detail preservation, respectively. The Decom-Net decomposes the low-light image into an illumination map and a reflectance map based on the Retinex theory. The Denoise-Net aims to suppress the noise in the reflectance map. Subsequently, the illumination map obtained by Decom-Net and the reflectance map obtained by Denoise-Net are sent to the Relight-Net to improve image contrast and reconstruct details.
![fig4](https://user-images.githubusercontent.com/86350392/141397881-334d4764-5fe0-4412-9e87-fef882089c53.jpeg)
The proposed Relight-Net architecture. The Relight-Net consists of two modules: Contrast Enhancement Module (CEM) and Detail Reconstruction Module (DRM). CEM uses spatial information for contrast enhancement and DRM uses frequency information to preserve image details.

# Pytorch
This is a Pytorch implementation of R2RNet.
## Requirements
1. Python 3.x 
2. Pytorch == 1.9.0 (We used torch.fft.fftn(ifftn) and torch.fft.rfftn(irfftn) in our code). 
## Dataset
We have fixed the image naming bugs, you can download the LSRW dataset from: https://pan.baidu.com/s/1XHWQAS0ZNrnCyZ-bq7MKvA (code: wmrr). 

(Note: Some outdoor image pairs are not pixel-to-pixel aligned, there may be some local offsets between two images. In fact, we think that this has a limited impact on the LLIE task. If you think it has a great impact on your task, please choose the appropriate training strategy and evaluation metric by yourself.)

## Pre-trained model
You can download pre-trained models from：https://pan.baidu.com/s/1fYBAvzCuuzmaFmDDAlsCWA (code: wmr1), then put the pre-trained models into Decom, Denoise, Relight, respectively. 

The pre-trained VGG-16 model can be downloaded from:https://pan.baidu.com/s/1kf1uLjLaAMbfji0fPZKtEQ (code: wmrr).
## Testing Usage
python predict.py
## Training Usage
python trian.py
# Reference
Code borrows heavily from https://github.com/aasharma90/RetinexNet_PyTorch.
