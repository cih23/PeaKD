"""
1 & 8-bit MPQ (Mixed Precision Quantizaion) on BERT
Authors:
- Tairen piao (piaotairen@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University
This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from torch.autograd import Function
import xnor_cuda



def Binarize(tensor):
    binarized = torch.where(tensor>0, torch.ones_like(tensor,dtype=torch.float32,device='cuda'), torch.full((tensor.shape),-1, dtype=torch.float32,device='cuda'))
    return binarized

def xnor_linear(input, weight,bias=True):

    # bin_input = xnor_cuda.encode_rows(input)
    weight_col = Binarize(weight.t())

    bin_weight = xnor_cuda.encode_cols(weight_col)
    # bin_weight = weight_col
    # print(bin_input.shape, bin_weight.shape)


    # output2 = xnor_cuda.my_gemm(input,bin_weight)
    # output3 = output2.to(dtype=torch.int32)
    print('jinlaile')
    output1 = input.matmul(weight.t())
    output2 = xnor_cuda.test_gemm(input,bin_weight)
    print(torch.equal(output1, output2))

    # input = torch.ones(input.size(0),int(input.size(1)/32)).to(device='cuda')
  
    # bin_weight = torch.randn(int(weight.size(1)/32),weight.size(0)).to(device='cuda')
    # weightt = weight
    # print(torch.nonzero(torch.isnan(weight)))
    # output2 = xnor_cuda.test_gemm(input,bin_weight)
    # output2 = input.matmul(weight.t())


    if bias is not None:
        output2 += bias
    ret = output2

    return ret

def xnor_linear_inference(input, weight, bias=True):
    output = xnor_cuda.test_gemm(input, weight)

    if bias is not None:
        output += bias
    ret = output

    return ret


class BinarizeLinear_test(nn.Module):

    def __init__(self, in_features, out_features, bias = True):
        super(BinarizeLinear_test, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.binarized_weight = nn.Parameter(torch.Tensor(out_features,in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):

        input.data = Binarize(input.data)
        with torch.no_grad():
            self.binarized_weight.data = Binarize(self.weight)
            # out = xnor_linear(input, self.weight, self.bias)
        out = nn.functional.linear(input, self.binarized_weight, self.bias)
        # else:
        #     input.data = input.data.half()
        #     self.weight.data = Binarize(self.weight.org)
        #     self.weight.data = self.weight.data.to(dtype=torch.half)
        #     self.bias.data = self.bias.data.half()
        #     out = nn.functional.linear(input, self.weight, self.bias)
        return out

class BinarizeLinear_inference(nn.Module):

    def __init__(self, in_features, out_features, bias = True):
        super(BinarizeLinear_inference, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        
        input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
            self.weight.org.requires_grad = True
        
        if self.weight.data.dtype == torch.float:
            self.weight.data = Binarize(self.weight.org)
            # out = xnor_linear(input, self.weight, self.bias)
            out = nn.functional.linear(input, self.weight, self.bias)
        # else:
        #     input.data = input.data.half()
        #     self.weight.data = Binarize(self.weight.org)
        #     self.weight.data = self.weight.data.to(dtype=torch.half)
        #     self.bias.data = self.bias.data.half()
        #     out = nn.functional.linear(input, self.weight, self.bias)
        return out

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
            self.weight.org.requires_grad = True
        
        if self.weight.data.dtype == torch.float:
            self.weight.data = Binarize(self.weight.org)
            # out = xnor_linear(input, self.weight, self.bias)
            out = nn.functional.linear(input, self.weight, self.bias)
        # else:
        #     input.data = input.data.half()
        #     self.weight.data = Binarize(self.weight.org)
        #     self.weight.data = self.weight.data.to(dtype=torch.half)
        #     self.bias.data = self.bias.data.half()
        #     out = nn.functional.linear(input, self.weight, self.bias)

        return out



def my_linear(input, weight, bias=None):
    # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
    """
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    # tens_ops = (input, weight)
    # if not torch.jit.is_scripting():
    #     if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
    #         return handle_torch_function(linear, tens_ops, input, weight, bias=bias)
#     bin_input = torch.zeros(int(input.size(0)),int(input.size(1)/32)).to(device="cuda:0",dtype=torch.int)
#     bin_weight =torch.zeros(int(weight.size(0)/32),int(weight.size(1))).to(device="cuda:0",dtype=torch.int)
    
    # bin_input = xnor_cuda.encode_rows(input)
    # bin_weight = xnor_cuda.encode_cols(weight)
    # output = xnor_cuda.test_gemm(input,weight)
    # s_t = time.time()
    # ret = torch.addmm(bias, input, weight.t())
    output = input.matmul(weight)

    if bias is not None:
        output += bias
    ret = output

    return ret


def quantization(input, bits):
    scale_max = 2**(bits-1)-1
    scale_min = -(2**(bits-1))


    pmax = input.max()
    pmin = input.min()
    scale = scale_max - scale_min
    q_scale = pmax - pmin
    
    quantized = torch.round((input - pmin)*(scale / q_scale) + scale_min)
    
#     quantized = torch.floor((input+(2**(bits)-1)) * (scale / q_scale))
    # print(quantized)
    
    qmax = quantized.max()
    qmin = quantized.min()
    scale_q = quantized.max() - quantized.min()
    scale_dq = pmax - pmin
    
    dequantized = (quantized - qmin)*(scale_dq / scale_q) + pmin
    
    return dequantized

class q_Linear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(q_Linear, self).__init__(*kargs, **kwargs)
        self.q_min = torch.nn.Parameter(torch.Tensor(1), requires_grad = False)
        self.q_max = torch.nn.Parameter(torch.Tensor(1), requires_grad = False)
        
    def forward(self, input):

        a = float(torch.min(self.weight.data))
        b = float(torch.max(self.weight.data))
        self.q_min.data = torch.Tensor([a]).to(device = 'cuda:0')
        self.q_max.data = torch.Tensor([b]).to(device = 'cuda:0')
            

        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data = quantization(self.weight.data,8)
        
        # if self.weight.data.dtype == torch.half:
        #     self.weight.data = self.weight.data.to(dtype=torch.half)

        out = nn.functional.linear(input, self.weight, self.bias)

        return out


class q_Linear_test(nn.Module):

    def __init__(self, in_features, out_features, bias = True):
        super(q_Linear_test, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.quantized_weight = nn.Parameter(torch.Tensor(out_features,in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):

        # input.data = Binarize(input.data)
        with torch.no_grad():
            self.quantized_weight.data = quantization(self.weight.data,8)
            # out = xnor_linear(input, self.weight, self.bias)
        out = nn.functional.linear(input, self.quantized_weight, self.bias)
        # else:
        #     input.data = input.data.half()
        #     self.weight.data = Binarize(self.weight.org)
        #     self.weight.data = self.weight.data.to(dtype=torch.half)
        #     self.bias.data = self.bias.data.half()
        #     out = nn.functional.linear(input, self.weight, self.bias)
        return out

class qEmbedding(nn.Embedding):

    def __init__(self, *kargs, **kwargs):
        super(qEmbedding, self).__init__(*kargs, **kwargs)
        self.q_min = torch.nn.Parameter(torch.Tensor(1), requires_grad = False)
        self.q_max = torch.nn.Parameter(torch.Tensor(1), requires_grad = False)


    def forward(self, input):
        
        a = float(torch.min(self.weight.data))
        b = float(torch.max(self.weight.data))
        self.q_min.data = torch.Tensor([a]).to(device = 'cuda:0')
        self.q_max.data = torch.Tensor([b]).to(device = 'cuda:0')

        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        # self.weight.data = torch.zeros_like(self.weight.data)
        self.weight.data=quantization(self.weight.data,8)

        return torch.nn.functional.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)