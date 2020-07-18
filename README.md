# Pruning-model-for-ARM-CortexM
I integrated two kinds of model pruning methods and porting to ARM-CortexM with CMSIS library.

## Requirement
1. Pytorch 1.5.0
2. ARM-CMSIS_5
3. NuMaker-PFM-M487(development board)
4. Ubuntu 16.04

## Pruning methods
1. Convolutional layers pruning of [this paper](https://arxiv.org/abs/1608.08710)
2. Fully-connected layer pruning of [this paper](https://arxiv.org/abs/1506.02626)
