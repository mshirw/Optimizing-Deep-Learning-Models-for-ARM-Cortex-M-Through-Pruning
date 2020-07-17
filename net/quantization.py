import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix
from .prune import MaskedLinear

def ShiftRight(quant_data,shift_bits):# only support int type        
    return np.right_shift(quant_data,shift_bits)

def ShiftLeft(quant_data,shift_bits):# only support int type        
    return np.left_shift(quant_data,shift_bits)

def Shift_Right_Bits(quant_data):
    min_quant_data = quant_data.min()
    max_quant_data = quant_data.max()
    #print(max(abs(min_quant_data),abs(max_quant_data)))
    Shift_right_bits = int(np.ceil(np.log2(max(abs(min_quant_data),abs(max_quant_data))))) - 7
    # Shift_right_bits = int(np.ceil(np.log2(max(abs(min_quant_data),abs(max_quant_data))))) - 8
    if Shift_right_bits < 0:
        Shift_right_bits = 0
    return Shift_right_bits #>> shift_bits

def FracBits(weight):    
    min_wt = weight.min()
    max_wt = weight.max()
    #find number of integer bits to represent this range
    int_bits = int(np.ceil(np.log2(max(abs(min_wt),abs(max_wt)))))     
    frac_bits = 7 - int_bits #remaining bits are fractional bits (1-bit for sign)

    return frac_bits

def Quantize(fp_data, frac_bits):
    #floating point weights are scaled and rounded to [-128,127], which are used in 
    #the fixed-point operations on the actual hardware (i.e., microcontroller)    
    quant_int8 = np.clip(np.round(fp_data*(2**frac_bits)), -128, 127)

    return quant_int8

def FracBits_torch(weight):
    min_wt = weight.min()
    max_wt = weight.max()
    int_bits = int(torch.ceil(torch.log2(max(abs(min_wt),abs(max_wt)))))
    frac_bits = 7 - int_bits

    return frac_bits

def Quantize_torch(weight, frac_bits):
    quant_int8 = torch.clamp(torch.round(weight*(2**frac_bits)), -128, 127)
    
    return quant_int8

def Quantize_2(data, bits):
    data_abs = abs(data)
    data_max = data_abs.max()
    factor = (2**bits - 1) / 2
    factor = factor / data_max
    data = np.round(data * factor)
    data = np.where(data == 128, 127, data)
    data = torch.from_numpy(data)
    print(data.min(), data.max())
    return data

def weight_quantization(model):
    for name, param in model.named_parameters():
        weight_q = Quantize_torch(param.cpu().detach(), FracBits_torch(param.cpu().detach()))
        param.data = weight_q        

    return model


