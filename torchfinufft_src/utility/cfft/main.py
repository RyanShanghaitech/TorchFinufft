import numpy
import torch
from numpy.typing import NDArray
from torch import Tensor
from typing import Iterable, Iterator

# both ifftshift() and fftshift() here is neccessary to make FFT consist with DFT

def fftnc(x:NDArray|Tensor, axes: Iterable|None=None) -> NDArray|Tensor:
    if axes is not None:
        if not isinstance(axes, Iterable): raise TypeError("")
        else: axes = tuple(axes)
    
    if isinstance(x, numpy.ndarray):
        x = numpy.fft.ifftshift(x, axes=axes)
        x = numpy.fft.fftn(x, axes=axes)
        x = numpy.fft.fftshift(x, axes=axes)
        return x

    if isinstance(x, torch.Tensor):
        x = torch.fft.ifftshift(x, dim=axes)
        x = torch.fft.fftn(x, dim=axes)
        x = torch.fft.fftshift(x, dim=axes)
        return x

    raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(x)!r}")

def ifftnc(x:NDArray|Tensor, axes: Iterable|None=None) -> NDArray|Tensor:
    if axes is not None:
        if not isinstance(axes, Iterable): raise TypeError("")
        else: axes = tuple(axes)
    
    if isinstance(x, numpy.ndarray):
        x = numpy.fft.ifftshift(x, axes=axes)
        x = numpy.fft.ifftn(x, axes=axes)
        x = numpy.fft.fftshift(x, axes=axes)
        return x

    if isinstance(x, torch.Tensor):
        x = torch.fft.ifftshift(x, dim=axes)
        x = torch.fft.ifftn(x, dim=axes)
        x = torch.fft.fftshift(x, dim=axes)
        return x
    
    raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(x)!r}")