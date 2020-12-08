#!/usr/bin/env python3
"""
Functions for running the 1D and 2D DFT and OS-CFAR

It is possible to run it both spiking and non-spiking
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
# Local libraries
import snn_dft_cfar.cfar
import snn_dft_cfar.dft
import snn_dft_cfar.utils.read_data
import snn_dft_cfar.utils.plot_tools


def dft_cfar(raw_data, dimensions, cfar_args, method="SNN"):
    """
    Call the routines for executing the DFT and the OS-CFAR
    """
    dft = snn_dft_cfar.dft.dft(raw_data, dimensions, method)
    cfar = run_cfar(dft, cfar_args, method)
    return dft, cfar


def run_cfar(dft_data, cfar_args, method="SNN"):
    """
    Run the corresponding OS-CFAR algorithm on the provided DFT data
    """
    if method=="numpy":
        cfar = snn_dft_cfar.cfar.OSCFAR(**cfar_args)
    elif method=="SNN":
        cfar = snn_dft_cfar.cfar.OSCFAR_SNN(**cfar_args)
    cfar(dft_data)
    return cfar


def plot(dft, cfar, dims, method):
    """
    Save figures containing the DFT and the CFAR of the experiment

    @param dft: Numpy array containing the Fourier transform result
    @param cfar: Numpy array containing the CFAR result
    @param dims: Number of dimensions. Used for generating the file name
    @param method: Method used for the algorithm, used for generating
    the title of the plot and the file name. {numpy | snn}
    """
    title = "{} DFT".format(method)
    # Obtain plot figures
    fig_dft = snn_dft_cfar.utils.plot_tools.plot_dft(dft, title, show=False)
    fig_cfar = snn_dft_cfar.utils.plot_tools.plot_cfar(cfar, show=False)
    # Save the figures to local files
    fig_dft.savefig("results/dft{}D_{}.eps".format(dims, method), dpi=150)
    fig_cfar.savefig("results/cfar{}D_{}.eps".format(dims, method), dpi=150)
    plt.show()
    return
