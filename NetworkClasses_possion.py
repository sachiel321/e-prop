# -*- coding: utf-8 -*-
"""
@author:yym
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scipy import signal
from ObjectClasses import Neuron, Spike
from ReservoirDefinitions import create_random_reservoir

thresh = 0.5
lens = 0.5
decay = 0.2
if_bias = True

################SNU model###################
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

    # @staticmethod
    # def backward(ctx, grad_h):
    #     z = ctx.saved_tensors
    #     s = torch.sigmoid(z[0])
    #     d_input = (1 - s) * s * grad_h
    #     return d_input

act_fun = ActFun.apply

def mem_update(ops, x, mem, spike, lateral = None):
    mem = mem * decay * (1. - spike) + ops(x)
    if lateral:
        mem += lateral(spike)
    spike = act_fun(mem)
    return mem, spike
############################################
###############LSM_SNU model################
class SNN(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, num_classes,possion_num=50,gpu='0'):
        super(SNN, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_classes = num_classes*4 #every class coded by 4 neurons
        self.hidden_size = hidden_size
        self.possion_num = possion_num
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias = if_bias)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes, bias = if_bias)
        self.device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")

        #monitor
        self.monitor_input = torch.zeros(self.batch_size, self.input_size, self.possion_num).to(self.device)
        self.monitor_fc1 = torch.zeros(self.batch_size, self.hidden_size, self.possion_num).to(self.device)
        self.monitor_fc2 = torch.zeros(self.batch_size, self.num_classes, self.possion_num).to(self.device)

    def forward(self, input, task, time_window):
        self.fc1 = self.fc1.float()
        self.fc2 = self.fc2.float()

        h1_mem = h1_spike = h1_sumspike = torch.zeros(self.batch_size, self.hidden_size).to(self.device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(self.batch_size, self.num_classes).to(self.device)

        for step in range(time_window):
            
            x = input
            
            sum_out = None

            for t in range(time_window):
                if task == "LSM":
                    x_t = torch.from_numpy(x[:,t]).float().to(self.device)
                elif task == 'STDP':
                    x_t = x[:,t].to(self.device)

                x_t = x_t.view(self.batch_size, -1)
                h1_mem, h1_spike = mem_update(self.fc1, x_t, h1_mem, h1_spike)
                #h1_sumspike += h1_spike
                
                h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
                #h2_sumspike += h2_spike
                with torch.no_grad():
                    self.monitor_fc1[:,:,t] = h1_spike.detach()
                    self.monitor_fc2[:,:,t] = h2_spike.detach()
                sum_out = h2_spike if sum_out is None else sum_out + h2_spike

        #outputs = h2_sumspike / time_window
        return sum_out

    def stdp_step(self,reward, lr):

        r_stdp(self.monitor_fc1[0],self.monitor_fc2[0],self.fc2.weight, reward, lr=lr)
        r_stdp(self.monitor_input[0],self.monitor_fc1[0],self.fc1.weight, reward, lr=lr)


class LSMNetwork:
    def __init__(self, dims, frac_inhibitory, w_matrix, fanout, 
                simulation_steps, num_in_ch, tau=20, t_ref=10, 
                propagation_time=10, ignore_frac=0.0,each_step_reset=True):
        #simulation_steps : total number of simulation steps to simulate time T in steps of dt = T/dt
        self.reset = each_step_reset
        self.ignore_frac = ignore_frac
        self.propagation_time = propagation_time
        self.tau = tau
        self.t_ref = t_ref
        self.dims = dims
        self.n_nodes = dims[0]*dims[1]*dims[2]
        self.num_in_ch = num_in_ch
        if num_in_ch<=self.n_nodes:
            mapped_nodes = torch.from_numpy(np.random.choice(self.n_nodes, size=num_in_ch, replace=False))
        else:
            mapped_nodes = torch.from_numpy(np.random.choice(self.n_nodes, size=num_in_ch, replace=True))
        self.mapped_nodes = mapped_nodes
        self.frac_inibitory = frac_inhibitory
        self.w_matrix = w_matrix
        self.fanout = fanout
        adj_mat, all_connections, all_weights = create_random_reservoir(dims, frac_inhibitory, w_matrix, fanout)
        #self.adj_mat = adj_mat
        self.all_connections = all_connections
        self.all_weights = all_weights
        self.neuronList = [Neuron(i, all_connections[i], all_weights[i], fanout, tau, t_ref, propagation_time) for i in range(len(all_connections))]
        self.simulation_steps = simulation_steps
        self.current_time_step = 0
        self.action_buffer = []
        #TODO:
        for i in range(simulation_steps):
            self.action_buffer.append([])
        return
    
    def add_input(self, input_spike_train):
        #input_spike_train : num_channels x simulation_steps matrix of all channels of the input spike train
        # for i in range(len(self.neuronList)):
        #     self.neuronList[i] = Neuron(i, self.all_connections[i], self.all_weights[i], self.fanout, self.tau, self.t_ref, self.propagation_time)
        for t_step in range(input_spike_train.shape[1]):
            self.action_buffer[t_step] = []
            for ch in range(self.num_in_ch):
                if input_spike_train[ch,t_step] > 0:
                    self.action_buffer[t_step].append((input_spike_train[ch,t_step], self.mapped_nodes[ch]))
        return
    
    def simulate(self):
        rate_coding = torch.zeros([self.n_nodes,self.simulation_steps])
        frac = self.ignore_frac
        for t_step in range(self.simulation_steps):
            #print(t_step)
            if len(self.action_buffer[t_step])>0:
                for action in self.action_buffer[t_step]:
                    spike_val = action[0]
                    target_node = action[1]
                    spike_produced = self.neuronList[int(target_node)].receive_spike(t_step, spike_val)
                    if spike_produced != None:
                        if t_step > frac*self.simulation_steps:
                            rate_coding[target_node][t_step] += 1
                        receiver_nodes = spike_produced.receiver_nodes
                        spike_values = spike_produced.spike_values
                        receive_times = spike_produced.receive_times
                        for node in range(len(receiver_nodes)):
                            if(receive_times[node]<self.simulation_steps):
                                self.action_buffer[int(receive_times[node])].append((int(spike_values[node]), receiver_nodes[node]))
        #if self.reset:
        #reset
        for i in range(len(self.neuronList)):
            self.neuronList[i].reset_spike()
        for step in range(self.simulation_steps):
            self.action_buffer[t_step] = []
        return rate_coding
############################################
