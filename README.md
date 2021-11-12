# R2RNet
Official code of "R2RNet: Low-light Image Enhancement via Real-low to Real-normal Network." Jiang Hai, Zhu Xuan, Ren Yang, Yutong Hao, Fengzhu Zou, Fang Lin, and Songchen Han(Submitted to IEEE transaction on Image Processing)

Paper link: https://arxiv.org/abs/2106.14501
## Network Architecture
![fig3](https://user-images.githubusercontent.com/86350392/141397717-abff2d12-e810-4744-96e2-a1ce6af87002.jpeg)
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
