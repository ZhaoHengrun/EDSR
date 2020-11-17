# EDSR
EDSR Baseline的pytorch复现 <br/>
参考自<br/>
https://arxiv.org/pdf/1707.02921.pdf <br/>
https://github.com/twtygqyy/pytorch-edsr <br/>
https://github.com/thstkdgus35/EDSR-PyTorch <br/>
## Requirements
Python 3.8<br/>
PyTorch 1.6.0<br/>
Numpy 1.19.2<br/>
Pillow 7.2.0<br/>
Visdom 0.1.8.9<br/>
Wandb 0.10.10<br/>
## Usage:
### Make datasets
所用的目录需要手动创建<br/>
使用`utils/`目录下的`generate_train.m`制作训练数据集<br/>
使用`utils/`目录下的`make_dataset.m`制作测试数据集	<br/>
生成的训练，测试数据集图片分别保存在`datasets/train`，`datasets/test`目录下	<br/>
### Train
运行`python train.py`进行训练	<br/>
模型保存在`checkpoints/`目录下	<br/>
### Test&Eval
运行`python calc_psnr&output.py`进行测试，生成的图片保存在`results/`目录下	<br/>
`calc_psnr&output.py`中写的psnr计算结果可能有问题，可使用`utils/`目录下的`compute_psnr.m`计算psnr	<br/>
