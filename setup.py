from setuptools import setup

_packages = \
[
    "torchfinufft", 
]

_package_dir = \
{
    "torchfinufft":"./torchfinufft_src/", 
}

setup\
(
    name = 'torchfinufft-ryan',
    packages = _packages,
    package_dir = _package_dir,
    include_package_data = True
)
