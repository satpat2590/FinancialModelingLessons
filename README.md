# (Self)-Guide to Financial Machine Learning

## Directory Structure

<b>data</b>: Used to store data for training models

<b>lib / lib64</b>: Libraries containing stored files for virtual environment usage

<b>src</b>: Containing C/C++ source files to be converted into shared objects usable by Python files 

<b>cppbuild.sh</b> Contains the command to "sync" the shared object to Python

<b>Makefile</b>: The Makefile to rebuild the C/C++ files into shared object files

    - Run "make" in the root directory and it should build the files

<b>mnist.py</b> Run this Python file to build a model using the MNIST dataset, which is capable of identifying numbers on pictures 

<b>setup.py</b> Python config file to identify the shared object file


## Financial Modeling

