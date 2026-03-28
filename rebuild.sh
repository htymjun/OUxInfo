echo "== Set compiler (GCC) =="
export CC=gcc
export CXX=g++

echo "== Clean previous builds =="
rm -rf build dist *.egg-info

echo "== Build wheel and sdist =="
python -m build

echo "== Reinstall wheel (force) =="
python -m pip install dist/*.whl --force-reinstall

