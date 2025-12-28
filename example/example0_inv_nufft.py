from numpy import *
from numpy.typing import *
from matplotlib.pyplot import *

import torch
import torch.nn as nn
from torch.types import Tensor, Size

from torchfinufft import *
from time import time
import slime
import mrarbgrad as mag
import mrarbdcf as mad

# parameters
useToeplitz = 1
usePrecond = 1
sDev = "cuda" if torch.cuda.is_available() else "cpu"
dev = torch.device(sDev)

# Get Shepp-Logan Phantom
nAx = 2; nPix = 256
arrPhant = slime.genPhant(nPix=nPix)
arrM0 = slime.Enum2M0(arrPhant)*slime.genPhMap(nPix=nPix)
arrM0 = torch.from_numpy(arrM0).to(dev, torch.complex64)

# Generate non-uniform trajectory
mag.setGoldAng(1)
_, lstArrG = mag.getG_Spiral(lNPix=nPix)
lstArrK = [mag.cvtGrad2Traj(arrG, 10e-6, 2.5e-6)[0] for arrG in lstArrG]

arrK = vstack(lstArrK).astype(float32)
arr2PiKT = 2*pi*arrK.T
arrDcf = mad.calDcf(nPix, arrK)

modNufft = Nufft(2, (nPix,)*nAx, Size(), arr2PiKT, dev)
with torch.no_grad():
    tenS0 = modNufft(arrM0)
    
if usePrecond:
    modLoss = ToeKspL2Loss(arrK, sqrt(arrDcf), (nPix,)*nAx, tenS0, dev)
else:
    modLoss = ToeKspL2Loss(arrK, sqrt(arrDcf).mean()*ones_like(arrDcf), (nPix,)*nAx, tenS0, dev)

# 3. Optimization (Inverse NUFFT)
tenM = torch.zeros((nPix,)*nAx, device=dev, dtype=torch.complex64, requires_grad=True)

optimizer = torch.optim.Adam([tenM], lr=0.1)
loss_fn = nn.MSELoss()

print("Starting Optimization...")
n = 100
t = time()
for i in range(n):
    optimizer.zero_grad()
    
    if useToeplitz:
        loss = modLoss(tenM)
    else:
        tenS = modNufft(tenM)
        loss = torch.mean(torch.abs(tenS - tenS0)**2)
    
    loss.backward()
    optimizer.step()
    
    if i % 1 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.6f}")
t = time() - t
print(f"Elapsed Time: {t:.3f}s")

# Visualization
figure(figsize=(12, 4))
subplot(131)
imshow(arrM0.abs().cpu(), cmap='gray')
title("Original")

subplot(132);
for i in range(len(lstArrK)): plot(*lstArrK[i].T[:nAx,:], ".-")
axis("equal")
title("K-space Trajectory")

subplot(133)
imshow(tenM.detach().abs().cpu(), cmap='gray')
title("Reconstructed")

show()