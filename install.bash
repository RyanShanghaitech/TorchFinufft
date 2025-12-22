# python -m build
rm -r *.egg-info build dist
pip uninstall torchfinufft-ryan -y
pip install .
rm -r *.egg-info build dist

# Note: if you don't delete this local `egg-info`, pip will think the package is right here, not in the site-package folder, the consequence will be: when uninstalling, since the package location is not in the site-package, the package can not be uninstalled because the package location is misled by `egg-info` thus it can't find the file to uninstall.

# that is to say, to uninstall package correctly, ensure its `egg_info` is not in the current folder, so it's wise to always remove `egg-info` after build.
