# NUFFT module for PyTorch

## Introduction
There is no built-in NUFFT function in PyTorch, neither is Toeplitz MSE loss module, which is important for non-Cartesian MRI reconstruction works. To fill this gap, this package provides:
1. A high performance NUFFT `torch.nn` module wrapping `cufinufft` [3] and `finufft` [1,2] - they are the fastest NUFFT backends to the best of my knowledge.
2. Another elegant MSE loss (mean square of l2 loss) module for non-Cartesian reconstruction with **DCF preconditioning** boosted by **Toeplitz** operator. Basically, this is done by replacing the two-pass NUFFTs with a Cartesian fast Fourier convolution. This method is also fast but slightly slower than cufinufft in practice. Use as you need.

Both CPU and GPU are supported. Benchmark indicates a 2ms (NUFFT module) or a 3ms (Toeplitz MSE loss module) time cost per iteration in a 256×256 inverse NUFFT problem using a RTX3090 GPU.

## Install
For offline installing:
```bash
$ bash install.bash
```
To install with pip:
```bash
$ pip install torchfinufft-ryan
```
Optionally, to enable CUDA computation, `cufinufft` has to be installed. To solve for the density compensation function, I recommend my `mrarbdcf` package.

## Usage
Please refer to the `exmaple` folder - there are minimal example(s) for tutorial. Please run:
```bash
$ pip install -r example/requirements.txt
```
to install essential packages for the examples.

## References
[1] Barnett AH. Aliasing error of the kerne exp(β√(1-z²)) in the nonuniform fast Fourier transform. Applied and Computational Harmonic Analysis. 2021 Mar 1;51:1–16.

[2] Barnett AH, Magland J, af Klinteberg L. A Parallel Nonuniform Fast Fourier Transform Library Based on an “Exponential of Semicircle" Kernel. SIAM J Sci Comput. 2019 Jan;41(5):C479–504.

[3] Shih Y hsuan, Wright G, Anden J, Blaschke J, Barnett AH. cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs. 2021 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW). 2021 June;688–97. 
