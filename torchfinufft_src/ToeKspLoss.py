import torch
from torch import Tensor, nn
from torch.autograd.function import FunctionCtx
import torch.nn.functional as F
from .utility import fftnc, ifftnc
import finufft, cufinufft

class ToePlan: pass

class ToeKspL2Loss(nn.Module):
    '''
    Calculate kspace l2-loss using Toeplitz kernel and DCF preconditioning
    Forward:
        ‖WFx-Wy‖²
        i.e.
        xᴴFᴴWᴴWFx - xᴴFᴴWᴴWy - yᴴWᴴWFx + yᴴWᴴWy
        note: the two middle terms are equivalent, not conjugate
    Backward:
        ∂‖WFx-Wy‖²/∂x
        i.e.
        2FᴴWᴴWFx - 2FᴴWᴴWy
    where F is NUFFT, x,y are vectors (typically are images and k-space groundtruth), W are density compensation function.
    For optimization (by Toeplitz operator replacement), we need the sampling pattern in F, and corresponding W for initialization.
    '''
    def __init__(self, tenK:Tensor, tenW:Tensor, tupSizeImg:tuple, tenS0:Tensor):
        '''
        :param tenK: k-space coordinate in `/pix`
        :type tenK: Tensor[nK,nAx]
        :param tenW: density compensation function
        :type tenW: Tensor[nK]
        :param tupSizeImg: image shape
        :type tupSizeImg: tuple[nAx]
        :param tenS0: k-space groundtruth
        :type tenS0: Tensor[nPass,nK]
        '''
        super().__init__()
        tenK = torch.as_tensor(tenK)
        tenW = torch.as_tensor(tenW)
        tenS0 = torch.as_tensor(tenS0)
        
        if tenW.shape[-1]!=tenS0.shape[-1]:
            raise AssertionError("tenW.shape[-1]!=tenS0.shape[-1]")
        
        if tenS0.ndim==1:
            tenW = tenW.unsqueeze(0)
            tenS0 = tenS0.unsqueeze(0)
        elif tenS0.ndim==2:
            pass
        else:
            raise AssertionError("tenS0.ndim")
        self.nPass = tenS0.shape[0]
        if tenW.shape[0]==1:
            tenW = tenW.repeat(self.nPass,1)
        self.nAx = len(tupSizeImg)
        self.tupSizeImg = tupSizeImg
        self.tupSizeImg_2x = tuple(2*dim-dim%2 for dim in tupSizeImg)
        
        self.pad = []
        for dim, dim_2x in zip(reversed(tupSizeImg), reversed(self.tupSizeImg_2x)):
            nPixPad = dim_2x - dim
            nPixPadFront = nPixPad // 2
            nPixPadBack = nPixPad - nPixPadFront
            self.pad += [nPixPadFront, nPixPadBack]
        self.pad = tuple(self.pad)
        
        # `tenKspApo`, apodization correspond to sampling convolution kernel
        if tenS0.is_cuda: fn=cufinufft
        elif tenS0.is_cpu: fn=finufft
        else: raise NotImplementedError("device")
        plan = fn.Plan(1, self.tupSizeImg_2x, self.nPass, dtype="complex64")
        ten2PiKT = (2*torch.pi)*tenK.T[:self.nAx]
        plan.setpts(*(ten2PiKT.contiguous().numpy() if fn==finufft else ten2PiKT))
        _tenW = tenW.contiguous().numpy() if fn==finufft else tenW.contiguous()
        tenImgKer:Tensor = torch.as_tensor(plan.execute(_tenW.conj()*_tenW), device=tenS0.device) # 2x
        tenKspApo:Tensor = fftnc(tenImgKer, 1+torch.arange(self.nAx)) # 2x
        self.register_buffer("tenKspApo", tenKspApo)
        
        # `tenFHWHWY`, FᴴWᴴWy
        _tenS0 = tenS0.contiguous().numpy() if fn==finufft else tenS0.contiguous()
        tenFHWHWY:Tensor = torch.as_tensor(plan.execute(_tenW.conj()*_tenW*_tenS0), device=tenS0.device) # 2x
        self.register_buffer("tenFHWHWY", tenFHWHWY)
        
        # `tenYHWHWY`, yᴴWᴴWy
        tenYHWHWY:Tensor = torch.linalg.vector_norm(tenW*tenS0) # 2x
        tenYHWHWY *= tenYHWHWY
        self.register_buffer("tenYHWHWY", tenYHWHWY)
    
    def forward(self, tenImg:Tensor):
        '''
        :param self: n.a.
        :param tenImg: Description
        :type tenImg: Tensor[nPass,nPix,...]
        '''
        if len(self.tupSizeImg)==tenImg.ndim:
            tenImg = tenImg.unsqueeze(0)
        if self.tupSizeImg!=tenImg.shape[1:] or self.nPass!=tenImg.shape[0]: # note: this loss function will not be a part of a model, and will only be used in training mode, in which batch number is a constant
            raise AssertionError("tenImg.shape")
        
        # self.tenImgZeroPad[...] = 0 # no need to reinit as long as the imsize is unchanged
        tenImg = F.pad(tenImg, self.pad)
        plan = ToePlan()
        plan.tenKspApo = self.tenKspApo
        plan.tenFHWHWY = self.tenFHWHWY
        plan.tenYHWHWY = self.tenYHWHWY
        return ToeKspL2LossAutogradFunc.apply(plan, tenImg)


class ToeKspL2LossAutogradFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx:FunctionCtx, plan:ToePlan, tenImgZeroPad:Tensor):
        '''
        :param ctx: n.a.
        :type ctx: FunctionCtx
        :param plan: includes
            - tenKspApo: kspace apodization corresponds to FᴴWᴴWF convolution
            - tenFHWHWY: FᴴWᴴWy
            - tenYHWHWY: yᴴWᴴWy
        :type plan: FunctionCtx
        :param tenImgZeroPad: zero-padded img
        :type tenImg: Tensor
        '''
        ctx.save_for_backward(tenImgZeroPad, plan.tenKspApo, plan.tenFHWHWY)
        ctx.tupSizeImg = tuple(dim//2+dim%2 for dim in tenImgZeroPad.shape[1:])
        ctx.nAx = tenImgZeroPad.ndim-1
        return torch.real(torch.sum(tenImgZeroPad.conj()*ifftnc(fftnc(tenImgZeroPad, 1+torch.arange(ctx.nAx))*plan.tenKspApo, 1+torch.arange(ctx.nAx))) - 2*torch.sum(tenImgZeroPad.conj()*plan.tenFHWHWY) + plan.tenYHWHWY)
    
    @staticmethod
    def backward(ctx:FunctionCtx, tenLoss:Tensor):
        '''
        :param ctx: includes
            - tenImgZeroPad: zero-padded img
            - tenKspApo: kspace apodization corresponds to FᴴWᴴWF convolution
            - tenFHWHWY: FᴴWᴴWy
        :type ctx: FunctionCtx
        :param tenLoss: scalar loss
        :type tenLoss: Tensor
        '''
        tenImgZeroPad, tenKspApo, tenFHWHWY = ctx.saved_tensors
        return None, tenLoss*2*(ifftnc(fftnc(tenImgZeroPad, 1+torch.arange(ctx.nAx))*tenKspApo, 1+torch.arange(ctx.nAx)) - tenFHWHWY)
        