from numpy import *
from matplotlib.pyplot import *

import torch
from torch import Tensor

import torchfinufft as tfn
from time import time
import mrphantom as mpt
import mrarbgrad as mag
import mrarbdcf as mad

# parameters
useToeplitz = 0
usePrecond = 1
# nAx = 2; fov = 0.384; nPix = 256; kTurbo = 1; nCh = 1; lamb = 0
# nAx = 2; fov = 0.384; nPix = 256; kTurbo = 16; nCh = 1; lamb = 0
# nAx = 2; fov = 0.384; nPix = 256; kTurbo = 16; nCh = 1; lamb = 1e-3
# nAx = 2; fov = 0.384; nPix = 256; kTurbo = 32; nCh = 1; lamb = 0
# nAx = 2; fov = 0.384; nPix = 256; kTurbo = 32; nCh = 1; lamb = 1e-3
nAx = 2; fov = 0.384; nPix = 256; kTurbo = 32; nCh = 2; lamb = 1e-3
sDev = "cuda" if torch.cuda.is_available() else "cpu"
# sDev = "cpu" # test
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
arrPhant = mpt.genPhant(nPix=nPix)
arrImg = mpt.Enum2M0(arrPhant)*mpt.genPhMap(nPix=nPix)
arrCsm = mpt.genCsm(nAx, nPix, nCh)
tenCsm = torch.as_tensor(arrCsm, dtype=complex, device=dev)
tenImg = torch.from_numpy(arrImg).to(dev, complex)

# Generate non-Cartesian trajectories
gamma = 42.5756e6
dtGrad, dtAdc = 10e-6, 2.5e-6
sLim = 50 * gamma*fov/nPix
gLim = 1/nPix/dtAdc
mag.setMagGradSamp(100000)
mag.setMagTrajSamp(100000)
lstArrGrad = mag.getG_VDSpiral_RT(nPix=nPix, sLim=sLim, gLim=gLim, kRhoPhi0=0.5/(nPix*pi), kRhoPhi1=0.5/(2*pi))[1]
nPE = len(lstArrGrad)
lstArrK = [mag.cvtGrad2Traj(arrGrad, dtGrad, dtAdc)[0] for arrGrad in lstArrGrad]
lstArrK = lstArrK[:nPE//kTurbo] # undersampling
tAcq = lstArrK[0].shape[0]*dtAdc
print(f"{nPE} x {tAcq*1e3:.2f} ms = {nPE*tAcq*1e3:.2f} ms")

arrK = vstack(lstArrK)
arr2PiKT = 2*pi*arrK.T

# construct torch modules
modNufft = tfn.Nufft(2, (nPix,)*nAx, nCh, device=dev, dtype=complex)
modNufft.setpts(arr2PiKT)
with torch.no_grad():
    tenS0:Tensor = modNufft(tenImg*tenCsm)
    
if usePrecond:
    arrDcf = hstack(mad.solve(nPix, lstArrK))
else:
    arrDcf = ones([arrK.shape[0]])
    
if nAx==2: arrDcf *= (pi/4) / arrDcf.sum()
elif nAx==3: arrDcf *= (pi/6) / arrDcf.sum()
tenDcf = torch.as_tensor(arrDcf, device=dev, dtype=complex)

modLoss = tfn.ToeKspMSELoss(arrK, tenDcf, (nPix,)*nAx, tenS0, dev, complex)

# Optimization
tenImg_ = torch.zeros((nPix,)*nAx, device=dev, dtype=complex, requires_grad=True)
optimizer = torch.optim.LBFGS([tenImg_], tolerance_grad=0, tolerance_change=0); nIter = 20
# optimizer = torch.optim.SGD([tenM_], lr=1e3); nIter = 1000
# optimizer = torch.optim.Adam([tenM_], lr=1e-1); nIter = 1000

loss0 = -1
lstLoss = []
lossMin = inf
with torch.no_grad():
    tenImgBest = tenImg_.detach().clone()
def closure():
    global loss0, lstLoss, lossMin, tenImgBest
    optimizer.zero_grad()
    
    if useToeplitz:
        loss = modLoss(tenImg_*tenCsm).mean()
    else:
        modNufft.setpts(arr2PiKT) # test
        tenS = modNufft(tenImg_*tenCsm)
        loss = torch.mean(torch.abs(tenDcf.sqrt()*(tenS - tenS0))**2)
    
    for iAx in range(nAx):
        tenDiff = torch.diff(tenImg_, dim=iAx).abs()
        loss += lamb*(tenDiff**2 + 1e-6).sqrt().mean()
    
    if loss0<0: loss0 = loss.item()
    loss *= 1e0/loss0
        
    if torch.isnan(loss):
        print(f"[WARN] loss==NaN, iter={len(lstLoss)}")
        raise ValueError("loss==NaN")
    
    if loss.item() <= lossMin:
        lossMin = loss.item()
        with torch.no_grad():
            tenImgBest = tenImg_.detach().clone()
    
    loss.backward()
    lstLoss += [loss.item()]
    return loss
    
t = time()
for i in range(nIter):
    try: optimizer.step(closure)
    except KeyboardInterrupt: break
    if i%1==0: print(f"iter {i}: loss {lstLoss[-1]:.3e}")
t = time() - t
print(f"Elapsed Time: {t:.3f}s")

with torch.no_grad():
    tenImg_ = tenImgBest

# Visualization
figure(figsize=(12, 6))

subplot(231)
imshow(abs(arrImg), cmap='gray')
clim(0,1)
title("Original")

subplot(232)
for i in range(len(lstArrK)): plot(*lstArrK[i].T[:nAx,:], ".-")
axis("equal")
title("K-space Trajectory")

subplot(233)
imshow(tenImg_.detach().abs().cpu(), cmap='gray')
clim(0,1)
title("Reconstructed")

subplot(212)
plot(array(lstLoss), ".-")
yscale("log")
title("Convergence")

tight_layout(h_pad=1.25, w_pad=1.00, rect=[0.01,0.01,0.99,0.99])

show()