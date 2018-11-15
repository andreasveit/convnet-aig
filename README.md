# Convolutional Networks with Adaptive Inference Graphs (ConvNet-AIG)

This repository contains a [PyTorch](http://pytorch.org/) implementation of the paper [Convolutional Networks with Adaptive Inference Graphs](https://arxiv.org/abs/1711.11503) presented at ECCV 2018. 

The code is based on the [PyTorch example for training ResNet on Imagenet](https://github.com/pytorch/examples/tree/master/imagenet).

## Table of Contents
0. [Introduction](#introduction)
0. [Usage](#usage)
0. [Citing](#citing)
0. [Requirements](#requirements)
0. [Contact](#contact)

## Introduction
Do convolutional networks really need a fixed feed-forward structure? What if, after identifying the high-level concept of an image, a network could move directly to a layer that can distinguish fine-grained differences? Currently, a network would first need to execute sometimes hundreds of intermediate layers that specialize in unrelated aspects. Ideally, the more a network already knows about an image, the better it should be at deciding which layer to compute next. 

Convolutional networks with adaptive inference graphs (ConvNet-AIG) can adaptively define their network topology conditioned on the input image. Following a high-level structure similar to residual networks (ResNets), ConvNet-AIG decides for each input image on the fly which layers are needed. In experiments on ImageNet we show that ConvNet-AIG learns distinct inference graphs for different categories.

## Usage
There are two training files. One for CIFAR-10 `train.py` and one for ImageNet `train_img.py`.

The network can be simply trained with `python train.py` or with optional arguments for different hyperparameters:
```sh
$ python train.py --expname {your experiment name}
```

For ImageNet the folder containing the dataset needs to be supplied

```sh
$ python train_img.py --expname {your experiment name} [imagenet-folder with train and val folders]
```

Training progress can be easily tracked with [visdom](https://github.com/facebookresearch/visdom) using the `--visdom` flag. It keeps track of the learning rate, loss, training and validation accuracy as well as the activation rates of the gates for each class.


By default the training code keeps track of the model with the highest performance on the validation set. Thus, after the model has converged, it can be directly evaluated on the test set as follows
```sh
$ python train.py --test --resume runs/{your experiment name}/model_best.pth.tar
```

## Requirements 
This implementation is developed for 

0. Python 3.6.5
0. PyTorch 0.3.1
0. CUDA 9.1

## Target Rate schedules  
To improve performance and memory efficiency, the target rates of early, last and downsampling layers can be fixed so as to always execute the layers. 
Specifically, for the results in the paper the following target rate schedules are used for ResNet 50:
[1, 1, 0.8, 1, t, t, t, 1, t, t, t, t, t, 1, 0.7, 1] for t in [0.4, 0.5, 0.6, 0.7]
For ResNet 101 the following rates can be used:
([1]* 8).extend([t] * 25) for t in [0.3, 0.5]

For compatibility to newer versions, please make a pull request.

## Citing
If you find this helps your research, please consider citing:

```
@conference{Veit2018,
title = {Convolutional Networks with Adaptive Inference Graphs},
author = {Andreas Veit and Serge Belongie},
year = {2018},
journal = {European Conference on Computer Vision (ECCV)},
}
```


## Contact
andreas at cs dot cornell dot edu 
