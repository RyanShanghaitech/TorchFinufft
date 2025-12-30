from numpy.typing import NDArray
import torch
from torch import Tensor, nn
from torch.autograd.function import FunctionCtx
import finufft, cufinufft
from finufft import Plan as cpuPlan
from cufinufft import Plan as cudaPlan

class DirPlan: pass

def _nufft(x:Tensor, plan:cpuPlan|cudaPlan, dev:torch.device):
    if isinstance(plan, cpuPlan):
        _x = x.contiguous().numpy()
    elif isinstance(plan, cudaPlan):
        _x = x.contiguous().cuda()
    else:
        raise TypeError("plan")
    y = plan.execute(_x)
    return torch.as_tensor(y, device=dev)

class DirKspL2Loss(nn.Module):
    def __init__(self, tenK:Tensor|NDArray, tenDcf:Tensor|NDArray, tupSizeImg:tuple, tenS0:Tensor|NDArray, dev:torch.device|str = "cuda"):
        '''
        Compute ‖WFx-Wy‖,
        
        where F is NUFFT, x,y are vectors (typically are images and k-space groundtruth), W are density compensation function.
        
        This is plain implementation of ‖WFx-Wy‖. For Toeplitz implementation, please refer to `ToeKspL2Loss`.
        
        :param tenK: k-space coordinate in `/pix`
        :type tenK: Tensor|NDArray[nK,nAx]
        :param tenDcf: density compensation function
        :type tenDcf: Tensor|NDArray[nK]
        :param tupSizeImg: image shape
        :type tupSizeImg: tuple[nAx]
        :param tenS0: k-space groundtruth
        :type tenS0: Tensor|NDArray[nPass,nK]
        :param dev: device 
        :type dev: device|str
        '''
        super().__init__()
        print("[WARN] This is a slow implementation only for reference. For Toeplitz boosted implementation please use `ToeKspL2Loss`.")
        tenK  = torch.as_tensor(tenK, device=dev)
        W  = torch.as_tensor(tenDcf, device=dev).sqrt()
        y = torch.as_tensor(tenS0, device=dev)

        if W.shape[-1] != y.shape[-1]:
            raise AssertionError("tenW.shape[-1]!=tenS0.shape[-1]")

        if y.ndim == 1:
            W = W.unsqueeze(0)
            y = y.unsqueeze(0)
        elif y.ndim == 2:
            W = W.unsqueeze(0)
        else:
            raise AssertionError("tenS0.ndim")
        self.nPass = y.shape[0]
        if W.shape[0] == 1:
            W = W.repeat(self.nPass, 1)
        self.nAx = len(tupSizeImg)
        self.tupSizeImg = tupSizeImg
        
        # store constants
        self.register_buffer("tenK", tenK)
        self.register_buffer("W", W)
        self.register_buffer("y", y)

    def forward(self, tenImg:Tensor):
        '''
        :param self: n.a.
        :param tenImg: Description
        :type tenImg: Tensor[nPass,nPix,...]
        '''
        if len(self.tupSizeImg) == tenImg.ndim:
            tenImg = tenImg.unsqueeze(0)
        if self.tupSizeImg != tenImg.shape[1:] or self.nPass != tenImg.shape[0]:
            raise AssertionError("tenImg.shape")

        plan = DirPlan()
        plan.tenK = self.tenK
        plan.W = self.W
        plan.y = self.y
        plan.tupSizeImg = self.tupSizeImg
        plan.nAx = self.nAx

        return DirKspSQL2LossAutogradFunc.apply(plan, tenImg).sqrt()

class DirKspSQL2LossAutogradFunc(torch.autograd.Function):
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
    def forward(ctx:FunctionCtx, plan:DirPlan, tenImg:Tensor):
        x:Tensor = tenImg
        tenK:Tensor = plan.tenK
        W:Tensor = plan.W
        y:Tensor = plan.y
        tupSizeImg:tuple = plan.tupSizeImg
        nAx = plan.nAx
        dev = tenImg.device
        
        if tenImg.is_cuda:
            fn = cufinufft
        elif tenImg.is_cpu:
            fn = finufft
        else:
            raise NotImplementedError("device")

        # prepare plan
        ten2PiKT = (2*torch.pi) * tenK.T[:nAx]
        _ten2PiKT = ten2PiKT.contiguous().numpy() if fn==finufft else ten2PiKT.contiguous().cuda()
        
        pFwd = fn.Plan(2, tupSizeImg, y.shape[0], dtype="complex64")
        pFwd.setpts(*_ten2PiKT)
        
        pBwd = fn.Plan(1, tupSizeImg, y.shape[0], dtype="complex64")
        pBwd.setpts(*_ten2PiKT)

        # xᴴFᴴWᴴWFx
        Fx = _nufft(x, pFwd, dev)
        WWFx = W.conj()*W*Fx
        FWWFx = _nufft(WWFx, pBwd, dev)
        xFWWFx = torch.sum(x.conj()*FWWFx)
        
        # xᴴFᴴWᴴWy
        WWy = W.conj()*W*y
        FWWy = _nufft(WWy, pBwd, dev)
        xFWWy = torch.sum(x.conj()*FWWy)
        
        # yᴴWᴴWy
        Wy = W*y
        yWWy = torch.sum(Wy.conj()*Wy)

        # save context
        ctx.tupSizeImg = tupSizeImg
        ctx.nAx = nAx
        ctx.save_for_backward(tenK, W, Fx, y)
        
        # print(xFWWFx)
        # print(- 2*xFWWy.real)
        # print(yWWy)
        # exit()

        return torch.real(xFWWFx - 2*xFWWy.real + yWWy)

    @staticmethod
    def backward(ctx:FunctionCtx, gradLoss:Tensor):
        tenK, W, Fx, y = ctx.saved_tensors
        tupSizeImg = ctx.tupSizeImg
        nAx = ctx.nAx
        dev = gradLoss.device

        if gradLoss.is_cuda:
            fn = cufinufft
        elif gradLoss.is_cpu:
            fn = finufft
        else:
            raise NotImplementedError("device")
        
        # prepare plan
        ten2PiKT = (2*torch.pi) * tenK.T[:nAx]
        _ten2PiKT = ten2PiKT.contiguous().numpy() if fn==finufft else ten2PiKT.contiguous().cuda()
        
        pBwd = fn.Plan(1, tupSizeImg, y.shape[0], dtype="complex64")
        pBwd.setpts(*_ten2PiKT)
        
        # FᴴWᴴWFx
        WWFx = W.conj()*W*Fx
        FWWFx = _nufft(WWFx, pBwd, dev)
        
        # FᴴWᴴWy
        WWy = W.conj()*W*y
        FWWy = _nufft(WWy, pBwd, dev)

        return None, gradLoss*2*(FWWFx - FWWy)