import os
import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torchvision import datasets, transforms

def log(filename, content):
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)


def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)


def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:25} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')

def print_conv(init_model, pruned_model):
    init_layer = []
    pruned_layer = []
    for init_module in init_model.children():
        if isinstance(init_module, nn.Sequential):
            for i in range(len(init_module)):
                init_layer.append(init_module[i])
        elif isinstance(init_module, nn.Conv2d) or isinstance(init_module, nn.Linear):
            init_layer.append(init_module)
            # init_layer.append(init_module)

    for pruned_module in pruned_model.children():
        if isinstance(pruned_module, nn.Sequential):
            for i in range(len(pruned_module)):
                pruned_layer.append(pruned_module[i])
        elif isinstance(pruned_module, nn.Conv2d) or isinstance(pruned_module, nn.Linear):
            pruned_layer.append(pruned_module)

    print("\n--- Conv pruning ---")
    for i in range(len(init_layer)):
        if isinstance(init_layer[i], torch.nn.Conv2d) and isinstance(pruned_layer[i], torch.nn.Conv2d):
            total_init = np.prod(init_layer[i].weight.data.size())
            total_pruned = np.prod(pruned_layer[i].weight.data.size())
            print("conv{}.weight = {}/{} | total_pruned = {} | pruned = {:.2f}%\n".format(i, total_pruned, total_init, (total_init - total_pruned), 100 - ((total_pruned/total_init)*100)))
            # print(init_layer[i].weight.data.size())
            # print(total_init)

def test(model, use_cuda=True):
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else 'cpu')
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=False, **kwargs)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy
