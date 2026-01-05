from numpy import *
from matplotlib.pyplot import *

import torch
from torch import Tensor

from torchfinufft import *
from time import time
from mrphantom import *
import mrarbgrad as mag

# parameters
useToeplitz = 0
nAx = 2; nPix = 256
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
tenM0 = torch.from_numpy(arrM0).to(dev, complex)

# Generate non-Cartesian trajectories
lstArrG = mag.getG_Spiral(nPix=nPix,)[1]
nPE = len(lstArrG)
lstArrK = [mag.cvtGrad2Traj(arrG, 10e-6, 2.5e-6)[0] for arrG in lstArrG]

arrK = vstack(lstArrK)
arr2PiKT = 2*pi*arrK.T

# construct torch modules
modNufft = Nufft(2, (nPix,)*nAx, 1, arr2PiKT, dev, complex)
with torch.no_grad():
    tenS0:Tensor = modNufft(tenM0)

modLoss = ToeKspMSELoss(arrK, None, (nPix,)*nAx, tenS0, dev, complex)

# Optimization
tenM = torch.zeros((nPix,)*nAx, device=dev, dtype=complex, requires_grad=True)

optimizer = torch.optim.SGD([tenM], lr=1e3); nIter = 1000

loss0 = -1; lstLoss = []
def closure():
    global loss0, lstLoss
    
    optimizer.zero_grad()
    
    if useToeplitz:
        loss = modLoss(tenM).mean()
    else:
        tenS = modNufft(tenM)
        loss = torch.mean(torch.abs(tenS - tenS0)**2)
    
    if loss0<0: loss0 = loss.item()
    loss *= 1e0/loss0
    
    lstLoss += [loss.item()]
    
    loss.backward()
    return loss
    
t = time()
for i in range(nIter):
    optimizer.step(closure)
    if i%10==0: print(f"iter {i}: loss {lstLoss[-1]:.3e}")
t = time() - t
print(f"Elapsed Time: {t:.3f}s")

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