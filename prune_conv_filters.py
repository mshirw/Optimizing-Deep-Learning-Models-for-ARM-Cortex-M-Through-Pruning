import torch
import numpy as np
import copy
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F

def prune_network(args, network):
    device = torch.device("cuda")
    # prune network
    network = prune_step(network, args.prune_layers, args.prune_channels, False, args)
    network = network.to(device)
    print("-*-"*10 + "\n\tPrune network\n" + "-*-"*10)
    print(network)
    return network

def prune_step(network, prune_layers, prune_channels, independent_prune_flag, args):
    # network = network.cpu()
    model_layers = []
    i = 0
    
    if args.dataset == "mnist" and args.model == "lenet-light":
        for module in network.children():
            model_layers.append(module)
        
    count = 0 # count for indexing 'prune_channels'
    conv_count = 1 # conv count for 'indexing_prune_layers'
    dim = 0 # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]
    residue = None # residue is need to prune by 'independent strategy'
    if args.dataset == "mnist" and args.model == "lenet-light":
        for i in range(len(model_layers)):
            if isinstance(model_layers[i], torch.nn.Conv2d):
                if dim == 1:
                    new_, residue = get_new_conv(model_layers[i], dim, channel_index, independent_prune_flag)
                    model_layers[i] = new_
                    dim ^= 1 # xor

                if 'conv%d'%conv_count in prune_layers:
                    # channel_index = which channel index will prune.
                    channel_index = get_channel_index(model_layers[i].weight.data, prune_channels[count], residue)
                    new_ = get_new_conv(model_layers[i], dim, channel_index, independent_prune_flag)
                    model_layers[i] = new_
                    dim ^= 1
                    count += 1
                else:
                    residue = None
                conv_count += 1
    elif args.dataset == "cifar10": # for alexnet and vgg
        for i in range(len(network.feature)):
            if isinstance(network.feature[i], torch.nn.Conv2d):
                if dim == 1:
                    new_, residue = get_new_conv(network.feature[i], dim, channel_index, independent_prune_flag)
                    network.feature[i] = new_
                    dim ^= 1 # xor

                if 'conv%d'%conv_count in prune_layers:                
                    #channel_index = 要pruned的channel index
                    channel_index = get_channel_index(network.feature[i].weight.data, prune_channels[count], residue)
                    new_ = get_new_conv(network.feature[i], dim, channel_index, independent_prune_flag)
                    network.feature[i] = new_
                    dim ^= 1
                    count += 1
                else:
                    residue = None
                conv_count += 1

    
    if args.dataset == "mnist" and args.model == "lenet-light":
        network.conv1 = model_layers[0]
        network.conv2 = model_layers[1]

    # update to check last conv layer pruned
    print("===Change the linear layers input.===")
    if 'conv2' in prune_layers and args.dataset == "mnist" and args.model == "lenet-light":
        network.fc1 = get_new_linear(network.fc1, InferenceForFc(network))
    elif 'conv4' in prune_layers and args.dataset == "cifar10" and args.model == "alexnet-light":
        network.classifier[0] = get_new_linear(network.classifier[0], InferenceForFc_cifar(network))
    elif 'conv6' in prune_layers and args.dataset == "cifar10" and args.model == "vgg-light":
        network.classifier[0] = get_new_linear(network.classifier[0], InferenceForFc_cifar(network))

    return network

def get_channel_index(kernel, num_elimination, residue=None):
    # get cadidate channel index for pruning
    ## 'residue' is needed for pruning by 'independent strategy'

    sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)
    if residue is not None:
        sum_of_kernel += torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)
    
    vals, args = torch.sort(sum_of_kernel)

    return args[:num_elimination].tolist()

def index_remove(tensor, dim, index, removed=False):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    size_ = list(tensor.size())
    new_size = tensor.size(dim) - len(index)
    size_[dim] = new_size
    new_size = size_

    select_index = list(set(range(tensor.size(dim))) - set(index))
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))
    
    if removed:
        return new_tensor, torch.index_select(tensor, dim, torch.tensor(index))

    return new_tensor

def index_remove_fc(tensor, dim, in_features_size, removed=False):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    size_ = list(tensor.size())
    new_size = tensor.size(dim) - in_features_size
    size_[dim] = new_size
    new_size = size_

    # select_index = list(set(range(tensor.size(dim))) - set(in_features_size))
    select_index = list(set(range(in_features_size)))
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))

    if removed:
        return new_tensor, torch.index_select(tensor, dim, torch.tensor(in_features_size))

    return new_tensor

def get_new_conv(conv, dim, channel_index, independent_prune_flag=False):
    if dim == 0:
        new_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                                   out_channels=int(conv.out_channels - len(channel_index)),
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation, bias=False)
        
        new_conv.weight.data = index_remove(conv.weight.data, dim, channel_index)
        # new_conv.bias.data = index_remove(conv.bias.data, dim, channel_index)

        return new_conv

    elif dim == 1:
        new_conv = torch.nn.Conv2d(in_channels=int(conv.in_channels - len(channel_index)),
                                   out_channels=conv.out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation, bias=False)
        
        new_weight = index_remove(conv.weight.data, dim, channel_index, independent_prune_flag)
        residue = None
        if independent_prune_flag:
            new_weight, residue = new_weight
        new_conv.weight.data = new_weight
        # new_conv.bias.data = conv.bias.data

        return new_conv, residue

def get_new_linear(linear, in_features_size):
    new_linear = torch.nn.Linear(in_features=in_features_size, out_features=linear.out_features, bias=False)    
    new_linear.weight.data = index_remove_fc(linear.weight.data, 1, in_features_size)
    return new_linear

def InferenceForFc(model):
    data = torch.randn(1,1,28,28)
    # Conv1
    x = model.conv1(data)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

    # Conv2
    x = model.conv2(x)        
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=(3, 3), stride=3)    
    # Fully-connected    
    x = x.flatten(1, -1)
    in_features_size = x.size()[1]

    return in_features_size

def InferenceForFc_cifar(model):
    data = torch.randn(1,3,32,32)
    x = data
    for i in range(len(model.feature)):
        x = model.feature[i](x)
    
    x = nn.AvgPool2d(2)(x)
    # Fully-connected
    x = torch.flatten(x, 1)
    in_features_size = x.size()[1]

    return in_features_size