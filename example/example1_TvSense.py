from numpy import *
from matplotlib.pyplot import *

import torch
from torch import Tensor

from torchfinufft import *
from time import time
from mrphantom import *
import mrarbgrad as mag
import mrarbdcf as mad

# parameters
useToeplitz = 1
usePrecond = 1
nAx = 2; nPix = 256; kTurbo = 2; nCh = 2; lamb = 1e-3
sDev = "cuda" if torch.cuda.is_available() else "cpu"
dev = torch.device(sDev)

if sDev=="cuda":
    complex = torch.complex64
    float = torch.float32
    scomplex = "complex64"
elif sDev=="cpu":
    complex = torch.complex128
    float = torch.float64
    scomplex = "complex128"
else:
    raise NotImplementedError("dev")

# generate slime phantom
random.seed(0)
arrPhant = genPhant(nPix=nPix)
arrM0 = Enum2M0(arrPhant)*genPhMap(nPix=nPix)
arrCsm = genCsm(nAx, nPix, nCh)
tenCsm = torch.as_tensor(arrCsm, dtype=complex, device=dev)
tenM0 = torch.from_numpy(arrM0).to(dev, complex)

# Generate non-Cartesian trajectories
mag.setGoldAng(0)
mag.setShuf(1)
lstArrG = mag.getG_Spiral(nPix=nPix, sLim=100*42.5756e6*0.256/256)[1]
nPE = len(lstArrG)
lstArrK = [mag.cvtGrad2Traj(arrG, 10e-6, 2.5e-6)[0] for arrG in lstArrG]
lstArrK = lstArrK[:nPE//kTurbo] # undersampling
tAcq = lstArrK[0].shape[0]*1e-5
print(f"{nPE} x {tAcq*1e3:.2f} ms = {nPE*tAcq*1e3:.2f} ms")

arrK = vstack(lstArrK)
arr2PiKT = 2*pi*arrK.T

# construct torch modules
modNufft = Nufft(2, (nPix,)*nAx, nCh, arr2PiKT, dev, complex)
with torch.no_grad():
    tenS0:Tensor = modNufft(tenM0*tenCsm)
    
if usePrecond:
    arrDcf = hstack(mad.sovDcf(nPix, lstArrK))
else:
    arrDcf = ones([arrK.shape[0]])
    
if nAx==2: arrDcf *= (pi/4) / arrDcf.sum()
elif nAx==3: arrDcf *= (pi/6) / arrDcf.sum()
tenDcf = torch.as_tensor(arrDcf, device=dev, dtype=complex)

modLoss = ToeKspMSELoss(arrK, tenDcf, (nPix,)*nAx, tenS0, dev, complex)

# Optimization
tenM = torch.zeros((nPix,)*nAx, device=dev, dtype=complex, requires_grad=True)

optimizer = torch.optim.SGD([tenM], lr=1e3); nIter = 1000

loss0 = -1
lstLoss = []
lossMin = inf
with torch.no_grad():
    tenMBest = tenM.detach().clone()
def closure():
    global loss0, lstLoss, lossMin, tenMBest
    
    optimizer.zero_grad()
    
    if useToeplitz:
        loss = modLoss(tenM*tenCsm).mean()
    else:
        tenS = modNufft(tenM*tenCsm)
        loss = torch.mean(torch.abs(tenDcf.sqrt()*(tenS - tenS0))**2)
    
    for iAx in range(nAx):
        tenDiff = torch.diff(tenM, dim=iAx).abs()
        loss += lamb*(tenDiff**2 + 1e-8).sqrt().mean()
    
    if loss0<0: loss0 = loss.item()
    loss *= 1e0/loss0
        
    if torch.isnan(loss):
        print(f"[WARN] loss==NaN, iter={len(lstLoss)}")
        raise ValueError("loss==NaN")
    
    if loss.item() <= lossMin:
        lossMin = loss.item()
        with torch.no_grad():
            tenMBest = tenM.detach().clone()
    
    lstLoss += [loss.item()]
    
    loss.backward()
    return loss
    
t = time()
for i in range(nIter):
    try: optimizer.step(closure)
    except ValueError: break
    except KeyboardInterrupt: break
    if i%10==0: print(f"iter {i}: loss {lstLoss[-1]:.3e}")
t = time() - t
print(f"Elapsed Time: {t:.3f}s")

with torch.no_grad():
    tenM = tenMBest

# Visualization
figure(figsize=(12,6), dpi=150)

subplot(231)
imshow(abs(arrM0), cmap='gray')
clim(0,1)
title("Original")

subplot(232);
for i in range(len(lstArrK)): plot(*lstArrK[i].T[:nAx,:], ".-")
axis("equal")
title("K-space Trajectory")

subplot(233)
imshow(tenM.detach().abs().cpu(), cmap='gray')
clim(0,1)
title("Reconstructed")

subplot(212)
plot(array(lstLoss), ".-")
yscale("log")
title("Convergence")

show()