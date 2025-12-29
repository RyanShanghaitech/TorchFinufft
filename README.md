# NUFFT module for PyTorch

## Introduction
There is no NUFFT function in PyTorch, neither is Toeplitz $\ell_2$-loss module, which is important for non-Cartesian MRI reconstruction works. To fill this gap, this package provides:
1. A high performance NUFFT `torch.nn` module wrapping `cufinufft` [3] and `finufft` [1,2] - they are the fastest NUFFT backend I have ever seen.
2. Another elegant $\ell_2$-loss module for non-Cartesian reconstruction with **DCF preconditioning** boosted by **Toeplitz** operator. Basically, this is done by replacing the two-pass NUFFTs with a Cartesian fast Fourier convolution. This method is also fast but slightly slower than cufinufft in practice. Use as you need.

Both CPU and GPU are supported. Benchmark indicates a $2\mathbf{ms}$ (NUFFT module) or a $3\mathbf{ms}$ (Toeplitz $\ell_2$-loss module) time cost per iteration in a ${256}\times{256}$ inverse NUFFT problem using a RTX3090 GPU.

## Install
For offline installing:
```bash
bash install.bash
```

To install with pip:
```bash
pip install torchfinufft-ryan
```

## Usage
Please refer to the `exmaple` folder - there are minimal example(s) for tutorial.

## References
[1] Barnett AH. Aliasing error of the kerne exp($\beta\sqrt{1-z^2}$) in the nonuniform fast Fourier transform. Applied and Computational Harmonic Analysis. 2021 Mar 1;51:1–16.

[2] Barnett AH, Magland J, af Klinteberg L. A Parallel Nonuniform Fast Fourier Transform Library Based on an “Exponential of Semicircle" Kernel. SIAM J Sci Comput. 2019 Jan;41(5):C479–504.

[3] Shih Y hsuan, Wright G, Anden J, Blaschke J, Barnett AH. cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs. 2021 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW). 2021 June;688–97. 
