import torch
import util
import copy
import numpy as np
import torch.nn as nn
from net.vgg import VGG
from net.quantization import weight_quantization, Quantize, Quantize_2, FracBits, Shift_Right_Bits, ShiftRight, Quantize_torch, FracBits_torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.nn.modules.module import Module
from net.prune import PruningModule, MaskedLinear
from net.models import LeNet_5_onnx

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)

no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else 'cpu')


output_shift_dict = {}
stats = {}
batch_count = 0
fc_size = 0

transform_test = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=False)

def test(model, quant=False):
    model.eval()
    test_loss = 0
    correct = 0
    flag = False
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if quant == True:
                data = Quantize_torch(data.cpu(), FracBits_torch(data.cpu()))
                data = data.to(device)
                
            output = vgg_quant_activate_inference(model, data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            count += 1

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

def Shiftright_activate(data, shift_right_bit):
    x = ShiftRight(data.cpu().detach().numpy().astype(int), shift_right_bit)
    x = torch.from_numpy(x.astype('double')).type(torch.FloatTensor)
    x = x.to(device)
    return x

def update_stats(stats, key, shift_bit):   
    if key not in stats:
        stats[key] = {"shift_bit": 0, "batch_count": 1}
    else:
        stats[key]["shift_bit"] += shift_bit
        stats[key]["batch_count"] += 1
    
    return stats

def vgg_quant_activate_inference(model, data):
    x = data
    conv_count = 0
    fc_count = 0
    global batch_count
    global stats
    global fc_size
    batch_count += 1
    
    for i in range(len(model.feature)):
        x = model.feature[i](x)
        shift_right_bit = Shift_Right_Bits(x.cpu().detach().numpy().astype(int))
        if i == 0 or i == 3 or i == 6 or i == 8 or i == 11 or i == 13:
            name = "CONV" + str(conv_count)
            conv_count += 1
            output_shift_dict.update({name: shift_right_bit})
            stats = update_stats(stats, name, shift_right_bit)

        x = Shiftright_activate(x, shift_right_bit)

    
    x = nn.AvgPool2d(2)(x)
    fc_size = x.cpu().detach().numpy().shape[1:]
    # print(x.size())

    x = torch.flatten(x, 1)

    for j in range(len(model.classifier)):
        x = model.classifier[j](x)
        shift_right_bit = Shift_Right_Bits(x.cpu().detach().numpy().astype(int))
        if isinstance(model.classifier[j], nn.Linear):
            name = "FC" + str(fc_count)
            fc_count += 1
            output_shift_dict.update({name: shift_right_bit})
            stats = update_stats(stats, name, shift_right_bit)
            x = Shiftright_activate(x, shift_right_bit)
    
    return x


model = torch.load("./saves/pruned_vgg_7850_7845_0_0_10_10_10_10.ptmodel")

print(model)
util.print_nonzeros(model)

print("--- Quantization ---")
model = weight_quantization(model)
model = model.to(device)
test(model, True)
print(stats)
for key in stats:
    output_shift_dict[key] = int(np.round(stats[key]["shift_bit"] / stats[key]["batch_count"]))

print(output_shift_dict)
conv_index = ['0', '3', '6', '8', '11', '13']
conv_output_channel = []

for i in range(len(model.feature)):
    if isinstance(model.feature[i], nn.Conv2d):
        conv_output_channel.append(model.feature[i].weight.size()[0])
        # print(model.feature[i].weight.size()[0])

count = 0
# ====== output weight ======
model_weight = open("cortexm_weight.h","w")
for m in output_shift_dict:
    model_weight.write("#define ")
    model_weight.write(str(m) + "_OUT_SHIFT ")
    model_weight.write(str(output_shift_dict[m]))
    model_weight.write("\n")

model_weight.write("\n")

for i in range(6):
    model_weight.write("#define ")
    model_weight.write("CONV" + str(i) + "_BIAS {")
    for k in range(int(conv_output_channel[i])):
        if k == int(conv_output_channel[i]) - 1:
            model_weight.write('0')
        else:
            model_weight.write("0, ")

    model_weight.write("}")
    model_weight.write("\n")

model_weight.write("#define ")
model_weight.write("FC0_BIAS {")
for i in range(10):
    if i == 9:
        model_weight.write('0')
    else:
        model_weight.write("0, ")

model_weight.write("}")
model_weight.write("\n")


model_weight.write("\n")

for k in conv_index:
    # weight_name = "feature." + k + ".module.weight"
    weight_name = "feature." + k + ".weight"
    conv1_weight = model.state_dict()[weight_name].cpu().numpy().astype(int)
    conv1_weight = conv1_weight.transpose(0, 2, 3, 1)
    print("weight_shape:{}".format(conv1_weight.shape))
    conv1_weight = conv1_weight.flatten().astype(int)

    model_weight.write("#define ")
    model_weight.write("CONV" + str(count) + "_WEIGHT {")
    count += 1

    for i in range(len(conv1_weight)):
        if i == len(conv1_weight) - 1:
            model_weight.write(str(conv1_weight[i]))
        else:
            model_weight.write(str(conv1_weight[i]) + ', ')

    model_weight.write("}")
    model_weight.write("\n")
    model_weight.write("\n")

fc_reshape = [10]
for i in range(len(fc_size)):
    fc_reshape.append(fc_size[i])

fc_weight = model.state_dict()["classifier.0.weight"].cpu().numpy().astype(int)
fc_weight = fc_weight.reshape(fc_reshape).transpose(0,2,3,1).reshape(10, fc_weight.shape[1])
print("fc_weight:{}".format(fc_weight.shape))
fc_weight = fc_weight.flatten().astype(int)
model_weight.write("#define ")
model_weight.write("FC0_WEIGHT {")
for j in range(len(fc_weight)):
    if j == len(fc_weight) - 1:
        model_weight.write(str(fc_weight[j]))
    else:
        model_weight.write(str(fc_weight[j]) + ', ')

model_weight.write("}")
model_weight.write("\n")
model_weight.close()
