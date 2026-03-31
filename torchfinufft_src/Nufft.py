from numpy import prod
from numpy.typing import NDArray
import torch
from torch import Tensor, Size
import torch.nn as nn
import finufft
try: import cufinufft
except: hasCufn = 0
else: hasCufn = 1
from finufft import Plan

class Nufft(nn.Module):
    def __init__(self, nufft_type:int, n_modes:tuple, n_trans:int=1, eps:float=1e-6, isign:int|None=None, device:torch.device|str="cuda", dtype:str|torch.dtype="complex64"):
        """
        init function of NUFFT PyTorch module. Here we will decide the basic parameter of the NUFFT.

        Args:
            nufft_type (int): 1: Non-Uniform -> Uniform (e.g. MRI reconstruction); 2: Uniform -> Non-Uniform (e.g. MRI simulation).
            n_modes (tuple): Matrix size.
            n_trans (int): Batch size. Defaults to 1.
            eps (float): FINUFFT precision.
            isign (int): Sign of Fourier index, -1: Fourier Transform; 1: Inverse Fourier Transform; None: Auto-decide by `nufft-type`. Defaults to None.
            device (torch.device | str, optional): PyTorch device. Defaults to "cuda".
            dtype (str | torch.dtype, optional): FINUFFT data dtype. "complex64" or "complex128". Defaults to "complex64".
        """
        super().__init__()
        if dtype in (torch.complex64, torch.float32):
            scomplex = "complex64"
        elif dtype in (torch.complex128, torch.float64):
            scomplex = "complex128"
        elif isinstance(dtype, str):
            scomplex = dtype
        else:
            raise NotImplementedError("dtype")
        
        nAx = len(n_modes)
        self.device = torch.device(device)
        
        if self.device.type=="cuda": fn=cufinufft
        elif self.device.type=="cpu": fn=finufft
        else: raise NotImplementedError("device")
        
        self.fwdPlan = fn.Plan(nufft_type, n_modes, n_trans, eps, isign if isign is not None else None, dtype=scomplex)
        self.bwdPlan = fn.Plan(3-nufft_type, n_modes, n_trans, eps, -isign if isign is not None else None, dtype=scomplex)
        
        self.fn = fn
        self.nAx = nAx
        
    def setpts(self, pts:Tensor|NDArray):
        """
        NOTE: this will also change the points of the backward plan, so after `setpts()` and `forward()`, you must call `backward()` before another `setpts()`, or `backward()` will use the wrong points.
        
        Set points for internal nufft plans.

        Args:
            pts (Tensor | NDArray): Sampling pattern in `[nAx,nK]`
        """
        pts = torch.as_tensor(pts).contiguous()
        
        if self.device.type=="cpu":
            _pts = pts.cpu().numpy()
        elif self.device.type=="cuda":
            _pts = pts.cuda() # (self.device)
            """
            NOTE: In current cufinufft version, `_pts` must be on "cuda:0", no matter what device you use.
            """
        else:
            raise NotImplementedError("device")
        
        self.fwdPlan.setpts(*_pts[:self.nAx,:])
        self.bwdPlan.setpts(*_pts[:self.nAx,:])
        
    def forward(self, x:Tensor):
        """
        Perform NUFFT on signal `x`

        Args:
            x (Tensor): Target signal. Shape: `[n_trans,nK]` when `nufft_type=1`; `[n_trans,*n_modes]` when `nufft_type=2`
        """
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
        
        _data = data.contiguous().numpy() if isinstance(fwdPlan, finufft.Plan) else data.contiguous()
        if nufft_type == 1:
            out = fwdPlan.execute(_data.reshape(n_trans,-1)).reshape(*batch_shape,*n_modes)
        else:
            out = fwdPlan.execute(_data.reshape(n_trans,*n_modes)).reshape(*batch_shape,-1)
        out = torch.as_tensor(out, device=data.device)
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