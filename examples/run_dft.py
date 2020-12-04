#!/usr/bin/env python3
"""
Functions for running the 1D and 2D DFT, both spiking and non-spiking
"""
# Standard libraries
import numpy as np
# Local libraries
import snn_dft_cfar.dft
import snn_dft_cfar.pipeline
import snn_dft_cfar.utils.plot_tools
import snn_dft_cfar.utils.read_data


def dft_1d(chirp, method="snn"):
    """
    Return NumPy array with 1-D DFT

    The DFT is composed by real values corresponding to the positive
    side of the spectrum.
    """
    if method=="numpy":
        dft = np.abs(np.fft.fft(chirp))
        adjusted_dft = np.abs(dft)[1:int(dft.size/2)]
    elif method=="snn":
        dft_1d = snn_dft_cfar.pipeline.pipeline(chirp, dimensions=1)
        adjusted_dft = snn_dft_cfar.dft.adjust_1dfft_data(dft_1d)
    return adjusted_dft

def dft_2d(data_cube, method="snn", title="Spiking DFT"):
    """
    Return NumPy array with 2-D DFT

    The DFT is composed by real values corresponding to the positive
    side of the spectrum for the range, and for both positive and
    negative values for the velocity.

    The velocity spectrum is centered in zero
    """
    if method=="numpy":
        adjusted_dft = snn_dft_cfar.dft.fft_2d(data_cube)
    elif method=="snn":
        dft_2d = snn_dft_cfar.pipeline.pipeline(data_cube, dimensions=2)
        adjusted_dft = snn_dft_cfar.dft.adjust_2dfft_data(dft_2d)
    return adjusted_dft


def main(dims=2, method="snn", title="Spiking DFT",
         filename="../data/BBM/scenario1/samples_ch_1_scenario1.txt"):
    # Only the 900 first samples contain information
    data_cube = snn_dft_cfar.utils.read_data.bbm_get_datacube(filename)[:, :900]
    if dims==1:
        chirp_n = 15
        dft = dft_1d(data_cube[chirp_n], method)
    if dims==2:
        dft = dft_2d(data_cube, method, title)
    fig = snn_dft_cfar.utils.plot_tools.plot_dft(dft, title)
    fig.savefig("results/dft{}D_scenario1_{}.eps".format(dims, method), dpi=150)

if __name__ == "__main__":
    main()
