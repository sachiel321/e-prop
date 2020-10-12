import math
import numpy as np
import warnings
import numbers
from typing import List, Tuple, Optional, overload

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.utils.rnn import _VF
from collections import namedtuple




################SNU model###################
thresh = 0.5
lens = 0.5
decay = 0.2

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

class LIF:
    def __init__(self, Vth=0.5, dt=0.001, Vres=0, tau_ref=4, tau_m = -1, Rm=1, Cm=10,Mode="E"):

        #simulation parameters
        self.dt = dt                         #(seconds)

        #LIF parameters
        self.mode = Mode
        self.Vres = Vres                     #resting potential (mV)
        self.Vm = self.Vres                  #initial potential (mV)
        self.t_rest = -1                     #initial resting time point
        self.tau_ref = tau_ref               #(ms) : refractory period
        self.Vth = Vth                       #(mV)

        self.Rm = Rm                         
        self.Cm = Cm                          
        if tau_m!=-1:
            self.tau_m = tau_m               #(ms)
        else:
            self.tau_m = self.Rm * self.Cm   #(ms)

        self.V_spike = Vth+0.5         #spike delta (mV)
            
    def update(self, I, time_stamp):
        if time_stamp > self.t_rest:
            self.Vm = self.Vm + (((I*self.Rm - self.Vm) / self.tau_m) * self.dt) #膜电位增长
            
            if self.Vm >= self.Vth:
                self.Vm = self.V_spike #发射脉冲
                self.t_rest = time_stamp + self.tau_ref #发射脉冲后重置不应期
        else:
            self.Vm = self.Vres #在不应期内
        if self.mode == "E":
            return self.Vm
        elif self.mode == "I":
            return -self.Vm
        else:
            print("Please set neuron mode.")
            return 0
    
    def initialize(self):
        self.Vm = self.Vres
        self.t_rest = -1

#########################################################################
class RNNCellBase(nn.Module):
    __constants__ = ['input_size', 'hidden_size', 'bias']

    input_size: int
    hidden_size: int
    bias: bool
    weight_ih: Tensor
    weight_hh: Tensor
    # WARNING: bias_ih and bias_hh purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(self, input_size: int, hidden_size: int, bias: bool, num_chunks: int) -> None:
        super(RNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(num_chunks * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(num_chunks * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(num_chunks * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input: Tensor) -> None:
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input: Tensor, hx: Tensor, hidden_label: str = '') -> None:
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)


class RNNCell(RNNCellBase):
    r"""An Elman RNN cell with tanh or ReLU non-linearity.

    .. math::

        h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})

    If :attr:`nonlinearity` is `'relu'`, then ReLU is used in place of tanh.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``

    Inputs: input, hidden
        - **input** of shape `(batch, input_size)`: tensor containing input features
        - **hidden** of shape `(batch, hidden_size)`: tensor containing the initial hidden
          state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: h'
        - **h'** of shape `(batch, hidden_size)`: tensor containing the next hidden state
          for each element in the batch

    Shape:
        - Input1: :math:`(N, H_{in})` tensor containing input features where
          :math:`H_{in}` = `input_size`
        - Input2: :math:`(N, H_{out})` tensor containing the initial hidden
          state for each element in the batch where :math:`H_{out}` = `hidden_size`
          Defaults to zero if not provided.
        - Output: :math:`(N, H_{out})` tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    Examples::

        >>> rnn = nn.RNNCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
                hx = rnn(input[i], hx)
                output.append(hx)
    """
    __constants__ = ['input_size', 'hidden_size', 'bias', 'nonlinearity']
    nonlinearity: str

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = "tanh") -> None:
        super(RNNCell, self).__init__(input_size, hidden_size, bias, num_chunks=1)
        self.nonlinearity = nonlinearity

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')
        if self.nonlinearity == "tanh":
            ret = _VF.rnn_tanh_cell(
                input, hx,
                self.weight_ih, self.weight_hh,
                self.bias_ih, self.bias_hh,
            )
        elif self.nonlinearity == "relu":
            ret = _VF.rnn_relu_cell(
                input, hx,
                self.weight_ih, self.weight_hh,
                self.bias_ih, self.bias_hh,
            )
        else:
            ret = input  # TODO: remove when jit supports exception flow
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))
        return ret


################################################

EligALIFStateTuple = namedtuple('EligALIFStateTuple', ('s', 'z', 'z_local', 'r'))

class EligALIF():
    def __init__(self, n_in, n_rec, tau=20., thr=0.03, dt=1., dtype=torch.float32, dampening_factor=0.3,
                 tau_adaptation=200., beta=1.6,
                 stop_z_gradients=False, n_refractory=1):

        if tau_adaptation is None: raise ValueError("alpha parameter for adaptive bias must be set")
        if beta is None: raise ValueError("beta parameter for adaptive bias must be set")

        self.n_refractory = n_refractory
        self.tau_adaptation = tau_adaptation
        self.beta = beta
        self.decay_b = np.exp(-dt / tau_adaptation)

        if np.isscalar(tau): tau = torch.ones(n_rec, dtype=dtype) * np.mean(tau)
        if np.isscalar(thr): thr = torch.ones(n_rec, dtype=dtype) * np.mean(thr)

        tau = torch.tensor(tau, dtype=dtype) #convert data type to torch
        dt = torch.tensor(dt, dtype=dtype)

        self.dampening_factor = dampening_factor
        self.stop_z_gradients = stop_z_gradients
        self.dt = dt
        self.n_in = n_in
        self.n_rec = n_rec
        self.data_type = dtype

        self._num_units = self.n_rec

        self.tau = tau
        self._decay = torch.exp(-dt / tau)
        self.thr = thr

        #InputWeights
        self.w_in_var = Parameter(torch.from_numpy(np.random.randn(n_in, n_rec) / np.sqrt(n_in), dtype=dtype))
        #self.w_in_val = tf.identity(self.w_in_var)
        self.w_in_val = self.w_in_var

        #RecWeights
        self.w_rec_var = Parameter(torch.from_numpy(np.random.randn(n_rec, n_rec) / np.sqrt(n_rec), dtype=dtype))

        self.recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
        self.w_rec_val = torch.where(self.recurrent_disconnect_mask, torch.zeros_like(self.w_rec_var),
                                    self.w_rec_var)  # Disconnect self-connection

        self.variable_list = [self.w_in_var, self.w_rec_var]
        self.built = True

    @property
    def state_size(self):
        return EligALIFStateTuple(s=torch.tensor((self.n_rec, 2)), z=self.n_rec, r=self.n_rec, z_local=self.n_rec)

    @property
    def output_size(self):
        return [self.n_rec, torch.tensor((self.n_rec, 2))]

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        s0 = torch.zeros(size=(batch_size, n_rec, 2), dtype=dtype)
        z0 = torch.zeros(size=(batch_size, n_rec), dtype=dtype)
        z_local0 = torch.zeros(size=(batch_size, n_rec), dtype=dtype)
        r0 = torch.zeros(size=(batch_size, n_rec), dtype=dtype)

        return EligALIFStateTuple(s=s0, z=z0, r=r0, z_local=z_local0)

    def compute_z(self, v, b):
        adaptive_thr = self.thr + b * self.beta
        v_scaled = (v - adaptive_thr) / self.thr

        ################待解决###########
        z = spikefunction(v_scaled, self.dampening_factor)
        ##################################
        z = z * 1 / self.dt
        return z

    def compute_v_relative_to_threshold_values(self,hidden_states):
        v = hidden_states[..., 0]
        b = hidden_states[..., 1]

        adaptive_thr = self.thr + b * self.beta
        v_scaled = (v - adaptive_thr) / self.thr
        return v_scaled

    def __call__(self, inputs, state, scope=None, dtype=torch.float32, stop_gradient=None):

        decay = self._decay
        z = state.z
        z_local = state.z_local
        s = state.s
        r = state.r
        v, b = s[..., 0], s[..., 1]

        # This stop_gradient allows computing e-prop with auto-diff.
        #
        # needed for correct auto-diff computation of gradient for threshold adaptation
        # stop_gradient: forward pass unchanged, gradient is blocked in the backward pass
        use_stop_gradient = stop_gradient if stop_gradient is not None else self.stop_z_gradients
        if use_stop_gradient:
            #stop gridient
            z = z.detach()

        new_b = self.decay_b * b + z_local # threshold update does not have to depend on the stopped-gradient-z, it's local
        i_t = torch.matmul(inputs, self.w_in_val) + torch.matmul(z, self.w_rec_val) # gradients are blocked in spike transmission
        I_reset = z * self.thr * self.dt
        new_v = decay * v + i_t - I_reset

        # Spike generation
        is_refractory = r > 0
        zeros_like_spikes = torch.zeros_like(z)
        new_z = torch.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, new_b))
        new_z_local = torch.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, new_b))
        new_r = r + self.n_refractory * new_z - 1

        new_r = torch.clamp(new_r, 0., float(self.n_refractory)).detach()
        new_s = torch.stack((new_v, new_b), axis=-1)

        new_state = EligALIFStateTuple(s=new_s, z=new_z, r=new_r, z_local=new_z_local)
        return [new_z, new_s], new_state

    def compute_eligibility_traces(self, v_scaled, z_pre, z_post, is_rec):

        n_neurons = z_post.size()[2]
        rho = self.decay_b
        beta = self.beta
        alpha = self._decay
        n_ref = self.n_refractory

        # everything should be time major
        #transpose
        z_pre = z_pre.permute(1, 0, 2)
        v_scaled = v_scaled.permute(1, 0, 2)
        z_post = z_post.permute(1, 0, 2)

        psi_no_ref = self.dampening_factor / self.thr * torch.max(0., 1. - torch.abs(v_scaled))

        update_refractory = lambda refractory_count, z_post:\
            torch.where(z_post > 0,torch.ones_like(refractory_count) * (n_ref - 1),torch.max(0, refractory_count - 1))

        refractory_count_init = torch.zeros_like(z_post[0], dtype=torch.int32)
        refractory_count = tf.scan(update_refractory, z_post[:-1], initializer=refractory_count_init)
        refractory_count = torch.cat(([refractory_count_init], refractory_count), axis=0)

        is_refractory = refractory_count > 0
        psi = torch.where(is_refractory, torch.zeros_like(psi_no_ref), psi_no_ref)

        update_epsilon_v = lambda epsilon_v, z_pre: alpha[None, None, :] * epsilon_v + z_pre[:, :, None]
        epsilon_v_zero = torch.ones((1, 1, n_neurons)) * z_pre[0][:, :, None]
        epsilon_v = tf.scan(update_epsilon_v, z_pre[1:], initializer=epsilon_v_zero, )
        epsilon_v = torch.cat(([epsilon_v_zero], epsilon_v), axis=0)

        update_epsilon_a = lambda epsilon_a, elems:\
                (rho - beta * elems['psi'][:, None, :]) * epsilon_a + elems['psi'][:, None, :] * elems['epsi']

        epsilon_a_zero = torch.zeros_like(epsilon_v[0])
        epsilon_a = tf.scan(fn=update_epsilon_a,
                            elems={'psi': psi[:-1], 'epsi': epsilon_v[:-1], 'previous_epsi':shift_by_one_time_step(epsilon_v[:-1])},
                            initializer=epsilon_a_zero)

        epsilon_a = torch.cat(([epsilon_a_zero], epsilon_a), axis=0)

        e_trace = psi[:, :, None, :] * (epsilon_v - beta * epsilon_a)

        # everything should be time major
        e_trace = e_trace.permute(1, 0, 2, 3)
        epsilon_v = epsilon_v.permute(1, 0, 2, 3)
        epsilon_a = epsilon_a.permute(1, 0, 2, 3)
        psi = psi.permute(1, 0, 2)

        if is_rec:
            identity_diag = torch.eye(n_neurons)[None, None, :, :]
            e_trace -= identity_diag * e_trace
            epsilon_v -= identity_diag * epsilon_v
            epsilon_a -= identity_diag * epsilon_a

        return e_trace, epsilon_v, epsilon_a, psi

    def compute_loss_gradient(self, learning_signal, z_pre, z_post, v_post, b_post,
                              decay_out=None,zero_on_diagonal=None):
        thr_post = self.thr + self.beta * b_post
        v_scaled = (v_post - thr_post) / self.thr

        e_trace, epsilon_v, epsilon_a, _ = self.compute_eligibility_traces(v_scaled, z_pre, z_post, zero_on_diagonal)

        if decay_out is not None:
            e_trace_time_major = e_trace.permute(1, 0, 2, 3)
            filtered_e_zero = torch.zeros_like(e_trace_time_major[0])
            filtering = lambda filtered_e, e: filtered_e * decay_out + e * (1 - decay_out)
            filtered_e = tf.scan(filtering, e_trace_time_major, initializer=filtered_e_zero)
            filtered_e = filtered_e.permute(1, 0, 2, 3)
            e_trace = filtered_e

        gradient = torch.einsum('btj,btij->ij', learning_signal, e_trace)
        return gradient, e_trace, epsilon_v, epsilon_a



def pseudo_derivative(v_scaled, dampening_factor):
    '''
    Define the pseudo derivative used to derive through spikes.
    :param v_scaled: scaled version of the voltage being 0 at threshold and -1 at rest
    :param dampening_factor: parameter that stabilizes learning
    :return:
    '''
    return torch.max(1 - torch.abs(v_scaled), 0) * dampening_factor


class SpikeFunction(torch.autograd.Function):

    @staticmethod
    def forward(v_scaled):
        '''
        The pytorch function which is defined as a Heaviside function (to compute the spikes),
        but with a gradient defined with the pseudo derivative.
        :param v_scaled: scaled version of the voltage being -1 at rest and 0 at the threshold
        :param dampening_factor: parameter to stabilize learning
        :return: the spike tensor
        '''
        z_ = torch.ge(torch.tensor(v_scaled),torch.tensor(0.)).float()

        return z_

    @staticmethod
    def backward(v_scaled, dampening_factor,dy):
        dE_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)
        dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled,
                torch.zeros_like(dampening_factor)]

spikefunction =  SpikeFunction.apply