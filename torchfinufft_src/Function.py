from numpy import prod
import torch
import torch.nn as nn
from torch.types import Tensor, Size
import cufinufft as cufn

class nufft(nn.Module):
    def __init__(self, nufft_type:int, n_modes:tuple, batch_shape:Size, pts:Tensor):
        super().__init__()
        
        nAx = len(n_modes)
        n_trans = prod(batch_shape).astype(int).item()
        self.register_buffer('pts', pts.contiguous())
        
        self.fwdPlan = cufn.Plan(nufft_type, n_modes, n_trans)
        self.bwdPlan = cufn.Plan(3-nufft_type, n_modes, n_trans)
        
        self.fwdPlan.setpts(*self.pts[:nAx,:])
        self.bwdPlan.setpts(*self.pts[:nAx,:])
        
    def forward(self, x:Tensor):
        return NufftAutogradFunc.apply(self.fwdPlan, self.bwdPlan, x)

class NufftAutogradFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fwdPlan:cufn.Plan, bwdPlan:cufn.Plan, data:Tensor):
        data = data.contiguous()
        
        nufft_type = fwdPlan.type
        n_modes = fwdPlan.n_modes
        n_trans = fwdPlan.n_trans
        
        nAx = len(n_modes)
        if nufft_type == 1: batch_shape = data.shape[:-1]
        else: batch_shape = data.shape[:-nAx]
        
        ctx.bwdPlan = bwdPlan
        
        if nufft_type == 1:
            out = fwdPlan.execute(data.reshape(n_trans,-1)).reshape(*batch_shape,*n_modes)
        else:
            out = fwdPlan.execute(data.reshape(n_trans,*n_modes)).reshape(*batch_shape,-1)
        return out

    @staticmethod
    def backward(ctx, data:Tensor):
        data = data.contiguous()
        
        bwdPlan = ctx.bwdPlan
        nufft_type = bwdPlan.type
        n_modes = bwdPlan.n_modes
        n_trans = bwdPlan.n_trans
        
        nAx = len(n_modes)
        if nufft_type == 1: batch_shape = data.shape[:-1]
        else: batch_shape = data.shape[:-nAx]
        
        if nufft_type == 1:
            out = bwdPlan.execute(data.reshape(n_trans,-1)).reshape(*batch_shape,*n_modes)
        else:
            out = bwdPlan.execute(data.reshape(n_trans,*n_modes)).reshape(*batch_shape,-1)
        return None, None, out