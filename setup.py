# setup.py
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='matmul',
    ext_modules=[
        CppExtension('matmul', ['src/matmul.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
