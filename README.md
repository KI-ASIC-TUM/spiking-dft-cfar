# spiking-DFT-CFAR

Implementation of a Spiking Neural Network solving the 2-D Discrete Fourier Transform and the OS-CFAR segmentation algorithm for automotive radar raw data


# Installation

Once you are in the main project directory, just install it as a normal Python3 package:

    pip install .


# Usage

The main module is the entry point for testing the algorithm. Create your custom configuration for the experiment or use the custom one, and select the number of dimensions (1 or 2) and the method to be used ("numpy" for the standard DFT and OS-CFAR, or "SNN" for the spiking version):

    usage: main.py [-h] [-d {1 2}] [-m {numpy SNN}] [-f] config_file

    positional arguments:
    config_file         Relative location of the configuration file

    optional arguments:
    -h, --help          show this help message and exit
    -d , --dimensions   {1 | 2} number of DFT dimensions
    -m , --method       {numpy | SNN} method used for running the system
    -f []               Get the S-DFT data from a local file
    -s []               Show the plot after the simulation

Example of usage:

    python3 main.py config config/scenario1_default.json --dimensions=1 --method=SNN -s

There is data containing a sample scenario with three targets and typical noise
sources in automotive radar. For more detailed specifications of the scene, go
to the _json_ file in `data/BBM/scenario1`.


# Experiment configuration

The parameters for the experiments are specified in a config file in json format.
There is a default configuration in the _config_ folder. Feel free to modify it
and try different simulation parameters:

* _cfar_args_: Standard OS-CFAR algorithm parameters. The guard and neighbour
cell parameters refer to half-window size. This means that `guarding_cells=3` will
create a window of 6 cells around the central cell for the 1D case, and a window
of 6x6 cells for the 2D case. _zero_padding_ indicates whether to calculate
the CFAR on the borders by adding zeroes around the frame
* _cfar_encoding_parameters_: Parameters for implementing time coding on the
provided input of the S-OSCFAR. NOTE: x_max is automatically re-calculated
within the code, based on the range of the input data
* _dft_encoding_parameters_: Paramters for implementing the rate coding of the
S-DFT.
