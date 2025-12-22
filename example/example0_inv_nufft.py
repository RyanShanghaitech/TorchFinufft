from numpy import *
from numpy.typing import *
from matplotlib.pyplot import *

import torch
import torch.nn as nn
from torch.types import Tensor, Size
from skimage.data import shepp_logan_phantom

from torchfinufft import *
from time import time
import slime
import mrarbgrad as mag

# 2. Setup Data and Simulate K-space
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get Shepp-Logan Phantom
nAx = 2; nPix = 256
arrPhant = slime.genPhant(nPix=nPix)
arrM0 = slime.Enum2M0(arrPhant)*slime.genPhMap(nPix=nPix)
arrM0 = torch.from_numpy(arrM0).to(device, torch.complex64)

# Generate non-uniform trajectory
_, lstArrG = mag.getG_Spiral(lNPix=nPix)
lstArrK = [mag.cvtGrad2Traj(arrG, 10e-6, 2.5e-6)[0] for arrG in lstArrG]

arr2PiKT = 2*pi*vstack(lstArrK).T.astype(float32)

modNufft2 = nufft2(2, (nPix,)*nAx, Size(), torch.from_numpy(arr2PiKT).to(device))
with torch.no_grad():
    kspace_data = modNufft2(arrM0)

# 3. Optimization (Inverse NUFFT)
recon_image = torch.zeros((nPix,)*nAx, device=device, dtype=torch.complex64, requires_grad=True)

optimizer = torch.optim.Adam([recon_image], lr=0.1)
loss_fn = nn.MSELoss()

print("Starting Optimization...")
n = 1000
t = time()
for i in range(n):
    if i%10==0: print(f"{i}/{n}")
    optimizer.zero_grad()
    
    kspace_pred = modNufft2(recon_image)
    loss = torch.mean(torch.abs(kspace_pred - kspace_data)**2)
    
    loss.backward()
    optimizer.step()
    
    if i % 50 == 0:
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
imshow(recon_image.detach().abs().cpu(), cmap='gray')
title("Reconstructed")

show()