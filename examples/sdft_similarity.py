#!/usr/bin/env python3
"""
Run the standard DFT and the S-DFT and print similarity measuremment
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
# Local libraries
import snn_dft_cfar.dft
import snn_dft_cfar.utils.read_data


def normalize(data):
    """
    Normalize the input data between 0 and 1
    """
    no_offset = data - data.min()
    normalized = no_offset / no_offset.max()
    return normalized

def rmse(data_1, data_2):
    """
    Calculate the Root Mean Square Error between the two input sets
    """
    error = data_1 - data_2
    rmse = np.sqrt((error**2).sum() / error.size)
    print("Root mean square error: {}".format(rmse))
    return rmse

def main(dimensions=2, plot=False):
    """
    Run the S-DFT and FFT and measure their similarity
    """
    dft_encoding_parameters = {
        "min_frequency": 1,
        "max_frequency": 50,
        "min_value": 0,
        "max_value": 2,
        "time_range": 1,
        "time_step": 0.0002
    }
    fname = "../data/BBM/scenario4/samples_ch_1_scenario4.txt"
    data_cube = snn_dft_cfar.utils.read_data.bbm_get_datacube(fname)[:, :900]
    if dimensions==1:
        chirp_n = 15
        raw_data = data_cube[chirp_n]
    if dimensions==2:
        raw_data = data_cube

    sdft = snn_dft_cfar.dft.dft(raw_data, dimensions, dft_encoding_parameters,
                                method="SNN")
    fft = snn_dft_cfar.dft.dft(raw_data, dimensions, dft_args=None,
                               method="numpy")
    if dimensions==2:
        fft = fft[:-1, :]

    # Normalize both outputs and calculate the RMSE
    sdft_norm = normalize(sdft)
    fft_norm = normalize(fft)
    rmse(sdft_norm, fft_norm)

    if plot:
        plt.subplot(2, 1, 1)
        plt.plot(sdft_norm)
        plt.subplot(2, 1, 2)
        plt.plot(fft_norm)
        plt.show()


if __name__ == "__main__":
    main()
