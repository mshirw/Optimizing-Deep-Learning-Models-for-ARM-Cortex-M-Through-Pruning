import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
from net.models import LeNet_5_onnx, LeNet_5_3x3
from net.quantization import weight_quantization, Quantize, FracBits, Quantize_2
from net.vgg import VGG
from net.alexnet import AlexNet
# from net.resnet_source import resnet18
import util
import copy
# from prune_conv_alex import prune_network, prune_step
# from prune_conv_vgg import prune_network, prune_step
# from prune_conv_mnist import prune_network, prune_step
from prune_conv_filters import prune_network, prune_step

os.makedirs('saves', exist_ok=True)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--retrain-epochs', type=int, default=60, metavar='N',
                    help='number of epochs to retrain (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--sensitivity', type=float, default=[2], nargs='+',
                    help="sensitivity value that is multiplied to layer's std in order to get threshold value")
parser.add_argument('--prune-layers', default=['conv1'], nargs='+',
                    help="prune layers of conv.")
parser.add_argument('--prune-channels', type=int, default=[1], nargs='+')

parser.add_argument('--no-train', type=bool, default=False,
                    help='disables training')
parser.add_argument('--load-path', type=str,
                    help='trained model load path to prune', default=None)
parser.add_argument('--dataset', type=str,
                    help='training dataset.', default='mnist')
parser.add_argument('--model', type=str,
                    help='training model.', default='LeNet')
parser.add_argument('--no-prune', type=bool, default=False,
                    help='disables pruning')

args = parser.parse_args()

#Parameter
lr_step = 250
factor = 0.1
Pruning = False
# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')

# Control Seed
torch.manual_seed(args.seed)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf) #extend numpy
if args.dataset == 'cifar10':
    print("--- Using dataset cifar10 ---")
elif args.dataset == 'mnist':
    print("--- Using dataset mnist ---")

if args.dataset == 'mnist':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(), 
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False)

elif args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=200,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False)

# Define which model to use
if args.dataset == 'mnist' and args.model == "lenet-light":
    model = LeNet_5_onnx().to(device)
elif args.dataset == 'cifar10' and args.model =="alexnet-light":
    # mask 會導致cmsis那邊的fc運算錯誤
    model = AlexNet().to(device)
elif args.dataset == 'cifar10' and args.model =="vgg-light":
    model = VGG(depth=7, bn=False, mask=False).to(device)

print(model)


# NOTE : `weight_decay` term denotes L2 regularization loss term
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

initial_optimizer_state_dict = optimizer.state_dict()

def test(model, quant=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if quant == True:
                data = Quantize(data.cpu(), FracBits(data.cpu()))
                data = data.to(device)                
                output = model(data)
            else:
                output = model(data)            
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


def train(model, epochs, finetune=False):
    model.train()
    best_top1 = 0
    for epoch in range(epochs):
        if epoch == lr_step:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
                print("current_learning_rate: ", param_group['lr'])

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            # data = data * 255
            optimizer.zero_grad()
            output = model(data)            
            loss = F.cross_entropy(output, target)
            loss.backward()

            if args.no_prune == False:
                # zero-out all the gradients corresponding to the pruned connections
                for name, p in model.named_parameters():
                    if 'mask' in name:
                        continue
                    tensor = p.data.cpu().numpy()
                    grad_tensor = p.grad.data.cpu().numpy()
                    grad_tensor = np.where(tensor==0, 0, grad_tensor)
                    p.grad.data = torch.from_numpy(grad_tensor).to(device)

            optimizer.step()
            
            if batch_idx % args.log_interval == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Epoch: {epoch} / {epochs} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)] Loss: {loss.item():.6f}')

        top1 = test(model)
        if top1 > best_top1:
            best_top1 = top1
            if finetune == False:
                print("--- The best accuracy at Epochs {} ---".format(epoch))
                torch.save(model, f"saves/initial_model_best.ptmodel")
            else:
                print("--- The best accuracy at Epochs {} ---".format(epoch))
                torch.save(model, f"saves/pruned_model_best.ptmodel")

def prune_by_std(model, s=0.25):
    """
    Note that `s` is a quality parameter / sensitivity value according to the paper.
    According to Song Han's previous paper (Learning both Weights and Connections for Efficient Neural Networks),
    'The pruning threshold is chosen as a quality parameter multiplied by the standard deviation of a layer’s weights'
    I tried multiple values and empirically, 0.25 matches the paper's compression rate and number of parameters.
    Note : In the paper, the authors used different sensitivity values for different layers.
    """
    count = 0
    layer_index = [0]
    for name, module in model.named_modules():
        if name in ['classifier.0', 'classifier.2', 'fc1']:
            threshold = np.std(module.weight.data.cpu().numpy()) * s[count]
            print(f'Pruning with threshold : {threshold} for layer {name}')
            prune(model, threshold, layer_index[count])
            count += 1

def prune(model, threshold, index):
    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            for i in range(len(layer)):
                if isinstance(layer[i], nn.Linear):
                    fc_layer = layer[i]
        elif isinstance(layer, nn.Linear):
            fc_layer = layer
        
    weight_dev = fc_layer.weight.device
    mask = nn.Parameter(torch.ones([fc_layer.weight.size()[0], fc_layer.weight.size()[1]]), requires_grad=False)
    mask_dev = fc_layer.weight.device
    # Convert Tensors to numpy and calculate
    tensor = fc_layer.weight.data.cpu().numpy()
    # mask = model.mask.data.cpu().numpy()
    new_mask = np.where(abs(tensor) < threshold, 0, mask)
    # Apply new weight and mask
    fc_layer.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
    mask = torch.from_numpy(new_mask).to(mask_dev)


# Initial training
if args.no_train:
    print("--- Load model ---")
    model = torch.load(args.load_path, map_location='cuda:0')
    model = model.to(device)
    accuracy = test(model)
    util.log(args.log, f"initial_accuracy {accuracy}")
    print("--- Before pruning ---")
    util.print_nonzeros(model)
    init_model = copy.deepcopy(model)
else:
    print("--- Initial training ---")
    util.print_nonzeros(model)
    train(model, args.epochs)
    Init_time = 0
    accuracy = test(model)
    torch.save(model, f"saves/initial_model.ptmodel")
    model = torch.load("saves/initial_model_best.ptmodel", map_location='cuda:0')
    util.log(args.log, f"initial_accuracy {accuracy}")
    print("--- Before pruning ---")
    util.print_nonzeros(model)
    init_model = copy.deepcopy(model)


if args.no_prune == False:
    model = model.to('cpu')
    model = prune_network(args, model)    
    print("--- pruning fc ---")
    prune_by_std(model, args.sensitivity)
    model = model.to(device)
    test(model)
    print("--- After pruning ---")
    util.print_nonzeros(model)
    util.print_conv(init_model, model)
    print("--- Retraining ---")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001) # Reset the optimizer because the model was changed.
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=factor)
    train(model, args.retrain_epochs, True)
    torch.save(model, f"saves/model_after_retraining.ptmodel")
    print("--- After Retraining ---")
    model = torch.load("saves/pruned_model_best.ptmodel", map_location='cuda:0')
    accuracy = test(model)

print("--- Quantization weight ---")
noquant_model = copy.deepcopy(model)
quant_model = weight_quantization(model)
quant_model = quant_model.cuda()
accuracy = test(quant_model, True)


"""
print("--- Model's state_dict ---")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
"""
