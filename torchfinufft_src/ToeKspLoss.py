from numpy.typing import NDArray
import torch
from torch import Tensor, nn
from torch.autograd.function import FunctionCtx
import torch.nn.functional as F
from .utility import fftnc, ifftnc
import finufft, cufinufft

class ToePlan: pass

class ToeKspL2Loss(nn.Module):
    def __init__(self, tenK:Tensor|NDArray, tenDcf:Tensor|NDArray, tupSizeImg:tuple, tenS0:Tensor|NDArray, dev:torch.device|str="cuda"):
        '''
        Calculate ‖WFx-Wy‖,
        
        where F is NUFFT, x,y are vectors (typically are images and k-space groundtruth), W are density compensation function.  
        
        For optimization (by Toeplitz operator replacement), we need the sampling pattern in F, and corresponding W for initialization.
            
        :param tenK: k-space coordinate in `/pix`
        :type tenK: Tensor|NDArray[nK,nAx]
        :param tenDcf: density compensation function
        :type tenDcf: Tensor|NDArray[nK]
        :param tupSizeImg: image shape
        :type tupSizeImg: tuple[nAx]
        :param tenS0: k-space groundtruth
        :type tenS0: Tensor|NDArray[nTran,nK]
        :param dev: device 
        :type dev: device|str
        '''
        super().__init__()
        tenK = torch.as_tensor(tenK, device=dev)
        W = torch.as_tensor(tenDcf, device=dev).sqrt()
        y = torch.as_tensor(tenS0, device=dev)
        
        if W.shape[-1]!=y.shape[-1]:
            raise AssertionError("tenW.shape[-1]!=tenS0.shape[-1]")
        
        if y.ndim==1:
            W = W.unsqueeze(0)
            y = y.unsqueeze(0)
        elif y.ndim==2:
            W = W.unsqueeze(0)
        else:
            raise AssertionError("tenS0.ndim")
        nTran = y.shape[0]
        if W.shape[0]==1:
            W = W.repeat(nTran,1)
        nAx = len(tupSizeImg)
        tupSizeImg = tupSizeImg
        tupSizeImg_2x = tuple(2*dim-dim%2 for dim in tupSizeImg)
        
        # `FWWF`
        if y.is_cuda: fn=cufinufft
        elif y.is_cpu: fn=finufft
        else: raise NotImplementedError("device")
        pBwd = fn.Plan(1, tupSizeImg_2x, nTran, dtype="complex64")
        ten2PiKT = (2*torch.pi)*tenK.T[:nAx]
        pBwd.setpts(*(ten2PiKT.contiguous().numpy() if fn==finufft else ten2PiKT.contiguous().cuda()))
        _W = W.contiguous().numpy() if fn==finufft else W.contiguous().cuda()
        FWWF_img:Tensor = pBwd.execute(_W.conj()*_W)
        FWWF_img = torch.as_tensor(FWWF_img, device=dev)
        FWWF_ksp:Tensor = fftnc(FWWF_img, 1+torch.arange(nAx))
        self.register_buffer("FWWF_ksp", FWWF_ksp)
        
        # FᴴWᴴWy
        _y = y.contiguous().numpy() if fn==finufft else y.contiguous().cuda()
        FWWy:Tensor = pBwd.execute(_W.conj()*_W*_y)
        FWWy = torch.as_tensor(FWWy, device=dev)
        self.register_buffer("FWWy", FWWy)
        
        # `yᴴWᴴWy
        Wy:Tensor = W*y # 2x
        yWWy = torch.sum(Wy.conj()*Wy, 1)
        self.register_buffer("yWWy", yWWy)
        
        # save context
        self.tupSizeImg = tupSizeImg
        self.nTran = nTran
        self.pad = []
        for dim, dim_2x in zip(reversed(tupSizeImg), reversed(tupSizeImg_2x)):
            nPixPad = dim_2x - dim
            nPixPadFront = nPixPad // 2
            nPixPadBack = nPixPad - nPixPadFront
            self.pad += [nPixPadFront, nPixPadBack]
        self.pad = tuple(self.pad)
    
    def forward(self, tenImg:Tensor):
        '''
        :param self: n.a.
        :param tenImg: Description
        :type tenImg: Tensor[nTran,nPix,...]
        '''
        tupSizeImg:tuple = self.tupSizeImg
        nTran:int = self.nTran
        
        if len(tupSizeImg)==tenImg.ndim:
            tenImg = tenImg.unsqueeze(0)
        if tupSizeImg!=tenImg.shape[1:] or nTran!=tenImg.shape[0]: # note: this loss function will not be a part of a model, and will only be used in training mode, in which batch number is a constant
            raise AssertionError("tenImg.shape")
        
        x = F.pad(tenImg, self.pad)
        plan = ToePlan()
        plan.FWWF_ksp = self.FWWF_ksp
        plan.FWWy = self.FWWy
        plan.yWWy = self.yWWy
        return ToeKspSQL2LossAutogradFunc.apply(plan, x).sqrt()


class ToeKspSQL2LossAutogradFunc(torch.autograd.Function):
    '''
    Forward:
        ‖WFx-Wy‖²
        i.e.
        xᴴFᴴWᴴWFx - xᴴFᴴWᴴWy - yᴴWᴴWFx + yᴴWᴴWy
        note: the two middle terms are equivalent, not conjugate
    Backward:
        ∂‖WFx-Wy‖²/∂x
        i.e.
        2FᴴWᴴWFx - 2FᴴWᴴWy
    '''
    @staticmethod
    def forward(ctx:FunctionCtx, plan:ToePlan, tenImgZeroPad:Tensor):
        x = tenImgZeroPad
        FWWF_ksp:Tensor = plan.FWWF_ksp
        FWWy:Tensor = plan.FWWy
        yWWy:Tensor = plan.yWWy
        nAx = tenImgZeroPad.ndim-1
        imaxes = (*range(1,nAx+1),)
        
        # xᴴFᴴWᴴWFx
        xFWWFx = torch.sum(x.conj()*ifftnc(FWWF_ksp*fftnc(x, imaxes), imaxes), imaxes)
        
        # xᴴFᴴWᴴWy
        xFWWy = torch.sum(x.conj()*FWWy, imaxes)
        
        # print(xFWWFx)
        # print(- 2*xFWWy.real)
        # print(yWWy)
        # exit()
        
        # save context
        ctx.save_for_backward(x, FWWF_ksp, FWWy)
        ctx.tupSizeImg = tuple(dim//2+dim%2 for dim in tenImgZeroPad.shape[1:])
        ctx.nAx = nAx
        
        return torch.real(xFWWFx - 2*xFWWy.real + yWWy)
    
    @staticmethod
    def backward(ctx:FunctionCtx, gradLoss:Tensor):
        x, FWWF_ksp, FWWy = ctx.saved_tensors
        nAx:int = ctx.nAx
        imaxes = (*range(1,nAx+1),)
        
        return None, gradLoss.reshape(gradLoss.shape+(1,)*nAx)*2*(ifftnc(fftnc(x, imaxes)*FWWF_ksp, imaxes) - FWWy)
        