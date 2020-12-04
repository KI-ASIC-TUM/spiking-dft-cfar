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


def dft_cfar(chirp, dimensions, cfar_args, method="SNN"):
    dft = snn_dft_cfar.dft.dft(chirp, dimensions, method)
    cfar = run_cfar(dft, cfar_args, method)
    return dft, cfar


def run_cfar(dft_data, cfar_args, method="SNN"):
    if method=="numpy":
        cfar = snn_dft_cfar.cfar.OSCFAR(**cfar_args)
    elif method=="SNN":
        cfar = snn_dft_cfar.cfar.OSCFAR_SNN(**cfar_args)
    cfar(dft_data)
    return cfar


def plot(dft, cfar, dims, method):
    title = "{} DFT".format(method)
    # Obtain plot figures
    fig_dft = snn_dft_cfar.utils.plot_tools.plot_dft(dft, title, show=False)
    fig_cfar = snn_dft_cfar.utils.plot_tools.plot_cfar(cfar, show=False)
    # Save the figures to local files
    fig_dft.savefig("results/dft{}D_{}.eps".format(dims, method), dpi=150)
    fig_cfar.savefig("results/cfar{}D_{}.eps".format(dims, method), dpi=150)
    plt.show()
    return


def get_cfar_args(dims, method):
    """
    Return default CFAR arguments
    """
    # Get default time-encoding parameters
    if method=="SNN":
        encoding_parameters = {
            "t_max": 50,
            "t_min": 0,
            "x_max": 100,
            "x_min": 0
        }
    else:
        encoding_parameters = {}

    # Get default CFAR parameters, based on number of dimensions
    if dims==1:
        cfar_args = {
            "scale_factor": 0.2,
            "guarding_cells": 6,
            "neighbour_cells": 15,
            "k": 6
        }
    else:
        cfar_args = {
            "scale_factor": 0.2,
            "guarding_cells": 3,
            "neighbour_cells": 4,
            "k": 8
        }
    cfar_args.update(encoding_parameters)
    return cfar_args


def main(dims=2, method="numpy",
         filename="../data/BBM/scenario1/samples_ch_1_scenario1.txt"):
    # Only the 900 first samples contain information
    data_cube = snn_dft_cfar.utils.read_data.bbm_get_datacube(filename)[:, :900]
    # Load CFAR parameters
    cfar_args = get_cfar_args(dims, method)
    # Run corresponding routine based on the number of dimensions
    if dims==1:
        chirp_n = 15
        raw_data = data_cube[chirp_n]
    if dims==2:
        raw_data = data_cube
    dft, cfar = dft_cfar(raw_data, dims, cfar_args, method)
    plot(dft, cfar, dims, method)
    return


if __name__ == "__main__":
    main()
