# spiking-DFT-CFAR

Implementation of a Spiking Neural Network solving the 2-D Discrete Fourier Transform and the OS-CFAR segmentation algorithm for automotive radar raw data


# Installation

Once you are in the main project directory, just install it as a normal Python3 package:

    pip install .


# Usage

The main module is the entry point for testing the algorithm. Create your custom configuration for the experiment or use the custom one, and select the number of dimensions (1 or 2) and the method to be used ("numpy" for the standard DFT and OS-CFAR, or "SNN" for the spiking version):

    python3 main.py config config/scenario1_default.json --dimensions=1 --method=SNN

