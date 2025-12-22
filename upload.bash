rm -rf dist; python -m build
python -m twine upload dist/*.tar.gz
rm -rf dist build *.egg-info