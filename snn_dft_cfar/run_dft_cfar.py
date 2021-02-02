#!/usr/bin/env python3
"""
Functions for running the 1D and 2D DFT and OS-CFAR

It is possible to run it both spiking and non-spiking
"""
# Standard libraries
import matplotlib.pyplot as plt
import logging
import numpy as np
import time
import pathlib
# Local libraries
import snn_dft_cfar.cfar
import snn_dft_cfar.dft
import snn_dft_cfar.utils.read_data
import snn_dft_cfar.utils.plot_tools

logger = logging.getLogger('S-DFT S-CFAR')


def dft_cfar(raw_data, dimensions, dft_args, cfar_args, method="SNN",
             from_file=False):
    """
    Call the routines for executing the DFT and the OS-CFAR
    """
    rpath = pathlib.Path(__file__).resolve().parent.parent.joinpath("results")
    fname = "{}/{}D-dft.txt".format(rpath, dimensions)
    if from_file:
        dft = np.loadtxt(fname)
    else:
        logger.info("Running DFT algorithm")
        t_0 = time.time()
        dft = snn_dft_cfar.dft.dft(raw_data, dimensions, dft_args, method)
        t_dft = time.time()
        np.savetxt(fname, dft)
    logger.info("Running CFAR algorithm")
    cfar = run_cfar(dft, cfar_args, method)
    t_cfar = time.time()

    logger.debug("Total DFT time: {:.5f}".format(t_dft-t_0))
    logger.debug("Total CFAR time: {:.5f}".format(t_cfar-t_dft))

    return dft, cfar


def run_cfar(dft_data, cfar_args, method="SNN"):
    """
    Run the corresponding OS-CFAR algorithm on the provided DFT data
    """
    if method=="numpy" or method=="ANN":
        cfar = snn_dft_cfar.cfar.OSCFAR(**cfar_args)
    elif method=="SNN":
        cfar = snn_dft_cfar.cfar.OSCFAR_SNN(**cfar_args)
    cfar(dft_data)
    return cfar


def plot(dft, cfar, dims, method, plot_together=True, show=True, fmt="pdf"):
    """
    Save figures containing the DFT and the CFAR of the experiment

    @param dft: Numpy array containing the Fourier transform result
    @param cfar: Numpy array containing the CFAR result
    @param dims: Number of dimensions. Used for generating the file name
    @param method: Method used for the algorithm, used for generating
    the title of the plot and the file name. {numpy | snn}
    @param plot_together: In the case of a 2D simulation, generate the
    DFT and CFAR plots in a single figure
    @param show: Show the resulting plots from the simulation
    """
    if method=="SNN":
        dft_title = "Spiking DFT"
        cfar_title = "Spiking OS-CFAR"
    else:
        dft_title = "Standard DFT"
        cfar_title = "Standard OS-CFAR"
    # Obtain plot figures
    results_path = pathlib.Path(__file__).parent.parent.joinpath("results")
    logger.info("Saving plots in {}".format(results_path))
    if not plot_together or dims==1:
        fig_dft = snn_dft_cfar.utils.plot_tools.plot_dft(dft, dft_title,
                                                         show=False)
        fig_cfar = snn_dft_cfar.utils.plot_tools.plot_cfar(cfar, cfar_title,
                                                           show=False)
        # Save the figures to local files
        fig_dft.savefig("{}/dft{}D_{}.{}".format(
                results_path,dims, method, fmt), dpi=150)
        fig_cfar.savefig("{}/cfar{}D_{}.{}".format(
                results_path, dims, method, fmt), dpi=150)
    else:
        fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
        plt.subplots_adjust(wspace=0.05)
        snn_dft_cfar.utils.plot_tools.plot_dft(dft, dft_title, show=False,
                                               ax=axes[0])
        snn_dft_cfar.utils.plot_tools.plot_cfar(cfar, cfar_title, show=False,
                                                ax=axes[1])
        fig.savefig("{}/pipeline{}D_{}.{}".format(
                results_path, dims, method, fmt), dpi=50)
    if show:
        plt.show()
    return
