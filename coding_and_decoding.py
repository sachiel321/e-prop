# -*- coding: utf-8 -*-
"""
@author:yym
"""
import torch
import torch.nn as nn
import math
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

######################coding method###################
def poisson_spike(t, f, dt=0.1, dim=1):
    """ Generate a Poisson spike train.

    t: length
    f: frequency
    dt: time step; default 0.1ms
    """
    if type(f) is torch.Tensor:
        f_temp = f.cpu().item()
    else:
        f_temp = f
    # dt, t in ms; f in Hz.
    return np.random.rand(dim, int(t / dt)) < (f_temp * dt / 10)

def poisson_spike_multi(t, f, dt=0.1, dim=1):
    """ Generate a Poisson spike train.

    t: length
    f: frequency
    dt: time step; default 0.1ms
    """
    temp_out = []

    if type(f) is torch.Tensor:
        f_temp = f.cpu().numpy()
    else:
        f_temp = f
    # dt, t in ms; f in Hz.
    for k in range(len(f)):
        #temp_out = np.random.rand(dim, int(t / dt)) < (f[k] * dt / 10) if temp_out is None else temp_out.append(np.random.rand(dim, int(t / dt)) < (f[k] * dt / 10))
         temp_out.append(np.random.rand(dim, int(t / dt)) < (f_temp[k] * dt / 10))
    return np.array(temp_out)

def coding_method(input,num,divisor): #coding
    out = None
    if type(input) == torch.Tensor:
        temp_input = input.clone().to(device)
    else:
        temp_input = input.copy()
    if temp_input > 0 :
        for j in range(num):
            temp_in = int(temp_input/divisor)
            temp_input -= temp_in * divisor
            divisor = divisor/10 
            temp_in = max(-9, min(9, temp_in))  
            i = temp_in + 9
            temp_out = torch.zeros([1,19])
            temp_out[0][i] = 1
            out = temp_out if out is None else torch.cat([out,temp_out],0)
    elif temp_input < 0:
    
        for j in range(num):
            temp_in = int(temp_input/divisor)
            temp_input -= temp_in * divisor
            divisor = divisor/10
            temp_in = max(-9, min(9, temp_in))  
            i = -temp_in + 9
            temp_out = torch.zeros([1,19])
            temp_out[0][i] = 1
            out = temp_out if out is None else torch.cat([out,temp_out],0)
    else:
        out = torch.zeros([num,19])
    return out.to(device)
    
def coding_method_multi(input,num,divisor): #coding
    out = None
    if type(input) == torch.Tensor:
        temp_input = input.clone().to(device)
    else:
        temp_input = input.copy()
    for k in range (len(temp_input)):
        temp_divisor = divisor
        if temp_input[k] > 0 :
            for j in range(num):
                temp_in = int(temp_input[k]/temp_divisor)
                temp_input[k] -= temp_in * temp_divisor
                temp_divisor = temp_divisor/10
                temp_in = max(-9, min(9, temp_in))  
                i = temp_in + 9
                temp_out = torch.zeros([1,19])
                temp_out[0][i] = 1
                out = temp_out if out is None else torch.cat([out,temp_out],0)
        elif temp_input[k] < 0:
            for j in range(num):
                temp_in = int(temp_input[k]/temp_divisor)
                temp_input[k] -= temp_in * temp_divisor
                temp_divisor = temp_divisor/10
                temp_in = max(-9, min(9, temp_in))  
                i = -temp_in + 9
                temp_out = torch.zeros([1,19])
                temp_out[0][i] = 1
                out = temp_out if out is None else torch.cat([out,temp_out],0)
        else:
            temp_out = torch.zeros([num,19])
            out = temp_out if out is None else torch.cat([out,temp_out],0)
    return out.to(device)

def coding_method_num(input,num,divisor):
    out = None
    if type(input) == torch.Tensor:
        temp_input = input.clone().to(device)
    elif type(input) == np.core.core.ndarray:
        temp_input = input.copy()
    else:
        temp_input = input
    if temp_input != 0:
        if temp_input < 0:
            temp_in = 1
            temp_out = torch.tensor([temp_in])
            temp_input = -temp_input
            out = temp_out if out is None else torch.cat([out,temp_out],0)
        else:
            temp_in = 0
            temp_out = torch.tensor([temp_in])
            out = temp_out if out is None else torch.cat([out,temp_out],0)
        for j in range(num):
            temp_in = int(temp_input/divisor)
            temp_input -= temp_in * divisor
            divisor = divisor/10
            temp_out = torch.tensor([temp_in])
            out = temp_out if out is None else torch.cat([out,temp_out],0)
    else:
        for j in range(num+1):
            temp_out = torch.tensor([0])
            out = temp_out if out is None else torch.cat([out,temp_out],0)
    return out.to(device)

def decode_num(input,divisor):
    decode_output = 0
    for count_decode in range(len(input)-1):
        decode_output = decode_output*10 + torch.clamp(input=input[count_decode+1],min=-9,max=9).item()
    decode_output /= divisor
    if input[0].item() > 5:
        decode_output = -decode_output
    return torch.tensor([decode_output]).to(device)

def decode_num_grad(input,divisor):
    decode_output = torch.zeros([1], requires_grad=True).to(device)
    for count_decode in range(len(input)-1):
        decode_output = decode_output*10 + torch.clamp(input=input[count_decode+1],min=-9,max=9)
    decode_output /= divisor
    if input[0].item() > 5:
        decode_output = -decode_output

    return decode_output

########################################################################
