from numpy import *
from matplotlib.pyplot import *

import torch
import torch.nn as nn

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
nAx = 2; nPix = 256; nCh = 8; kTurbo = 2
random.seed(42)
arrPhant = slime.genPhant(nPix=nPix)
arrM0 = slime.Enum2M0(arrPhant)*slime.genPhMap(nPix=nPix)
arrCsm = slime.genCsm(nAx, nPix, nCh)
tenCsm = torch.as_tensor(arrCsm, dtype=torch.complex64, device=dev)
tenM0 = torch.from_numpy(arrM0).to(dev, torch.complex64)

# Generate non-Cartesian trajectories
mag.setGoldAng(1)
_, lstArrG = mag.getG_Spiral(lNPix=nPix)
lstArrK = [mag.cvtGrad2Traj(arrG, 10e-6, 2.5e-6)[0] for arrG in lstArrG]
lstArrK = lstArrK[:len(lstArrK)//kTurbo] # undersampling

arrK = vstack(lstArrK).astype(float32)
arr2PiKT = 2*pi*arrK.T

# construct torch modules
modNufft = Nufft(2, (nPix,)*nAx, nCh, arr2PiKT, dev)
with torch.no_grad():
    tenS0 = modNufft(tenM0*tenCsm)
    
if usePrecond:
    arrDcf = mad.calDcf(nPix, arrK[:,:nAx]).astype(complex64)
else:
    arrDcf = ones([arrK.shape[0]]).astype(complex64)
    
# modLoss = DirKspL2Loss(arrK, arrDcf, (nPix,)*nAx, tenS0, dev)
modLoss = ToeKspL2Loss(arrK, arrDcf, (nPix,)*nAx, tenS0, dev)

# 3. Optimization (Inverse NUFFT)
tenM = torch.zeros((nPix,)*nAx, device=dev, dtype=torch.complex64, requires_grad=True)
# with torch.no_grad(): # test
#     tenM[:] = tenM0

optimizer = torch.optim.Adam([tenM], lr=1e-1)
loss_fn = nn.MSELoss()

n = 1000
t = time()
for i in range(n):
    optimizer.zero_grad()
    
    if useToeplitz:
        loss = modLoss(tenM*tenCsm).sum()
    else:
        tenS = modNufft(tenM*tenCsm)
        loss = torch.sum(torch.abs(tenS - tenS0)**2).sqrt()
    
    loss.backward()
    optimizer.step()
    
    if i % (n//10) == 0:
        print(f"Iteration {i}, Loss: {loss.item():.6f}")
t = time() - t
print(f"Elapsed Time: {t:.3f}s")

# Visualization
figure(figsize=(12, 4))
subplot(131)
imshow(abs(arrM0), cmap='gray')
title("Original")

subplot(132);
for i in range(len(lstArrK)): plot(*lstArrK[i].T[:nAx,:], ".-")
axis("equal")
title("K-space Trajectory")

subplot(133)
imshow(tenM.detach().abs().cpu(), cmap='gray')
title("Reconstructed")

show()