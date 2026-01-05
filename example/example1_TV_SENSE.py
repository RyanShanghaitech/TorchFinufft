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
# nAx = 2; nPix = 256; kTurbo = 16; nCh = 1; lamb = 0
nAx = 2; nPix = 256; kTurbo = 16; nCh = 8; lamb = 1e-3
sDev = "cuda" if torch.cuda.is_available() else "cpu"
dev = torch.device(sDev)

# generate slime phantom
random.seed(42)
arrPhant = genPhant(nPix=nPix)
arrM0 = Enum2M0(arrPhant)*genPhMap(nPix=nPix)
arrCsm = genCsm(nAx, nPix, nCh)
tenCsm = torch.as_tensor(arrCsm, dtype=torch.complex64, device=dev)
tenM0 = torch.from_numpy(arrM0).to(dev, torch.complex64)

# Generate non-Cartesian trajectories
mag.setGoldAng(0)
mag.setShuf(1)
lstArrG = mag.getG_VarDenSpiral(nPix=nPix, kRhoPhi0=0.5/(256*pi), kRhoPhi1=0.5/(4*pi))[1]
nPE = len(lstArrG)
lstArrK = [mag.cvtGrad2Traj(arrG, 10e-6, 2.5e-6)[0] for arrG in lstArrG]
lstArrK = lstArrK[:nPE//kTurbo] # undersampling

arrK = vstack(lstArrK).astype(float32)
arr2PiKT = 2*pi*arrK.T

# construct torch modules
modNufft = Nufft(2, (nPix,)*nAx, nCh, arr2PiKT, dev)
with torch.no_grad():
    tenS0:Tensor = modNufft(tenM0*tenCsm)
    
if usePrecond:
    arrDcf = hstack(mad.sovDcf(nPix, lstArrK)).astype(complex64)
else:
    arrDcf = ones([arrK.shape[0]]).astype(complex64)
    
if nAx==2: arrDcf *= (pi/4) / arrDcf.sum()
elif nAx==3: arrDcf *= (pi/6) / arrDcf.sum()
tenDcf = torch.as_tensor(arrDcf)

modLoss = ToeKspMSELoss(arrK, arrDcf, (nPix,)*nAx, tenS0, dev)

# Optimization
tenM = torch.zeros((nPix,)*nAx, device=dev, dtype=torch.complex64, requires_grad=True)

# optimizer = torch.optim.SGD([tenM], lr=1e3); nIter = 1000
# optimizer = torch.optim.LBFGS([tenM], lr=1e-1); nIter = 100
optimizer = torch.optim.Adam([tenM], lr=1e-1); nIter = 1000 # superior

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
figure(figsize=(12, 6))

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