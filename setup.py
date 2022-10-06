from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='numpy-nn',
    version='0.1.0',
    packages=find_packages(where='numpy_nn'),
    package_dir={'': 'numpy_nn'},
    py_modules=[splitext(basename(path))[0] for path in glob('numpy_nn/*.py')],
)
