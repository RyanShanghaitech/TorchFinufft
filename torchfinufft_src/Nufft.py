from numpy import prod
from numpy.typing import NDArray
import torch
from torch import Tensor, Size
import torch.nn as nn
import cufinufft, finufft
from finufft import Plan

class Nufft(nn.Module):
    def __init__(self, nufft_type:int, n_modes:tuple, n_trans:int, pts:Tensor|NDArray, dev:torch.device|str="cuda", dtype:torch.dtype=torch.float32):
        super().__init__()
        if dtype in (torch.complex64, torch.float32):
            complex = torch.complex64
            float = torch.float32
            scomplex = "complex64"
        elif dtype in (torch.complex128, torch.float64):
            complex = torch.complex128
            float = torch.float64
            scomplex = "complex128"
        else:
            raise NotImplementedError("dtype")
        pts = torch.as_tensor(pts, device=dev, dtype=float)
        
        nAx = len(n_modes)
        
        if pts.is_cuda: fn=cufinufft
        elif pts.is_cpu: fn=finufft
        else: raise NotImplementedError("device")
        
        self.fwdPlan = fn.Plan(nufft_type, n_modes, n_trans, dtype=scomplex)
        self.bwdPlan = fn.Plan(3-nufft_type, n_modes, n_trans, dtype=scomplex)
        
        self.fn = fn
        self.nAx = nAx
        self.setpts(pts)
        
    def setpts(self, pts:Tensor|NDArray):
        pts = torch.as_tensor(pts)
        
        _pts = pts.contiguous().numpy() if self.fn==finufft else pts.contiguous().cuda()
        self.fwdPlan.setpts(*_pts[:self.nAx,:])
        self.bwdPlan.setpts(*_pts[:self.nAx,:])
        
    def forward(self, x:Tensor):
        return NufftAutogradFunc.apply(self.fwdPlan, self.bwdPlan, x)

class NufftAutogradFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fwdPlan:Plan, bwdPlan:Plan, data:Tensor):
        nufft_type = fwdPlan.type
        n_modes = fwdPlan.n_modes
        n_trans = fwdPlan.n_trans
        
        nAx = len(n_modes)
        if nufft_type == 1: batch_shape = data.shape[:-1]
        else: batch_shape = data.shape[:-nAx]
        
        ctx.bwdPlan = bwdPlan
        
        _data = data.contiguous().numpy() if isinstance(bwdPlan, finufft.Plan) else data.contiguous()
        if nufft_type == 1:
            out = fwdPlan.execute(_data.reshape(n_trans,-1)).reshape(*batch_shape,*n_modes)
        else:
            out = fwdPlan.execute(_data.reshape(n_trans,*n_modes)).reshape(*batch_shape,-1)
        out = torch.as_tensor(out)
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
        
        _data = data.contiguous().numpy() if isinstance(bwdPlan, finufft.Plan) else data.contiguous()
        if nufft_type == 1:
            out = bwdPlan.execute(_data.reshape(n_trans,-1)).reshape(*batch_shape,*n_modes)
        else:
            out = bwdPlan.execute(_data.reshape(n_trans,*n_modes)).reshape(*batch_shape,-1)
        out = torch.as_tensor(out, device=data.device)
        return None, None, out