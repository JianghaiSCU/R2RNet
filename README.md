# R2RNet
Official code of "R2RNet: Low-light Image Enhancement via Real-low to Real-normal Network." Jiang Hai, Zhu Xuan, Ren Yang, Yutong Hao, Fengzhu Zou, Fang Lin, and Songchen Han(Submitted to IEEE transaction on Image Processing)

Paper link: https://arxiv.org/abs/2106.14501
## Network Architecture
![image](https://user-images.githubusercontent.com/86350392/123072534-382ae080-d448-11eb-856c-8086578a308e.png)
# Pytorch
This is a Pytorch implementation of R2RNet.
## Requirements
1. Python 3.x 
2. Pytorch 1.x.0
## Dataset
You can download the LSRW dataset from: https://pan.baidu.com/s/1sprnEO4F9z_ota4FLn8Q0Q (code: Wmr1)

If you use our code and  dataset, please cite our paper.
## Pre-trained model
We will release the pre-trained models as soon as possible.

You shold download the VGG model (https://pan.baidu.com/s/1Rn2NwHt9eZgfg6hQP-DrlQ code:wmr1)and put it into ./model.
## Testing Usage
python predict.py
## Training Usage
python trian.py
# Reference
Code borrows heavily from https://github.com/aasharma90/RetinexNet_PyTorch.
