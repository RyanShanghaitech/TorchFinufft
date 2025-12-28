# NUFFT module for PyTorch

This package includes:
1. A high performance NUFFT `torch.nn` module based on cufinufft [3] and finufft [1,2].
2. An ultra fast $\ell_2$-loss module for non-Cartesian reconstruction boosted by Toeplitz operator (basically by replacing the two-pass NUFFTs with a Cartesian fast Fourier convolution)

[1] Barnett AH. Aliasing error of the kerne exp($\beta\sqrt{1-z^2}$) in the nonuniform fast Fourier transform. Applied and Computational Harmonic Analysis. 2021 Mar 1;51:1–16.  
[2] Barnett AH, Magland J, af Klinteberg L. A Parallel Nonuniform Fast Fourier Transform Library Based on an “Exponential of Semicircle" Kernel. SIAM J Sci Comput. 2019 Jan;41(5):C479–504.  
[3] Shih Y hsuan, Wright G, Anden J, Blaschke J, Barnett AH. cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs. 2021 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW). 2021 June;688–97. 
