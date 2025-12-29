from numpy import *
from numpy.typing import NDArray
from matplotlib.pyplot import *
from matplotlib.figure import SubFigure, Figure

def ndimshow(img:NDArray, fig:SubFigure|Figure, figH:float=3):
    nAx = img.ndim
    s = img.shape
    imgAbs = abs(img)
    imgPh = angle(img)
    vmean, vstd = imgAbs.mean(), imgAbs.std()
    vmin, vmax = vmean-1*vstd, vmean+3*vstd
    if nAx==2:
        fig.set_size_inches(2*figH,1*figH)
        ax = fig.add_subplot(121)
        ax.imshow(imgAbs, cmap="gray", vmin=vmin, vmax=vmax)
        ax.axis("off")
        ax = fig.add_subplot(122)
        ax.imshow(imgPh, cmap="hsv", vmin=-pi, vmax=pi)
        ax.axis("off")
    elif nAx==3:
        fig.set_size_inches(3*figH,1*figH)
        ax = fig.add_subplot(131)
        ax.imshow(imgAbs[s[0]//2,:,:], cmap="gray", vmin=vmin, vmax=vmax)
        ax.axis("off")
        ax = fig.add_subplot(132)
        ax.imshow(imgAbs[:,s[1]//2,:], cmap="gray", vmin=vmin, vmax=vmax)
        ax.axis("off")
        ax = fig.add_subplot(133)
        ax.imshow(imgAbs[:,:,s[2]//2], cmap="gray", vmin=vmin, vmax=vmax)
        ax.axis("off")
    fig.tight_layout(pad=0)