#!/usr/bin/env python3
"""
Main description of the module.
"""
# Standard libraries
import numpy as np
# Local libraries
import snn_dft_cfar.dft
import snn_dft_cfar.pipeline
import snn_dft_cfar.utils.plot_tools
import snn_dft_cfar.utils.read_data


def fft_2d(data_cube):
    # Perform 2D FFT and get output dimensions for shaping the result
    data_fft = np.fft.fft2(data_cube)
    width, height = data_fft.shape
    modulus = np.abs(data_fft)
    # Remove negative values from the range spectrum. They have no
    # physical meaning.
    positive_range = modulus[:, 0:int(height/2)]
    # Re-adjust the plot so the velocity spectrum is centered around zero
    centered = np.copy(positive_range)
    centered[:int(width/2), :] = positive_range[int(width/2):,:]
    centered[int(width/2):, :] = positive_range[:int(width/2),:]
    # Place the speed on the horizontal axis, and range in the vertical one
    adjusted_data = np.flip(np.transpose(centered), axis=0)
    return adjusted_data

def dft_1d(chirp, method="snn", title="Spiking DFT"):
    if method=="numpy":
        dft = np.abs(np.fft.fft(chirp))
        adjusted_data = np.abs(dft)[1:450]
    elif method=="snn":
        dft_1d = snn_dft_cfar.pipeline.pipeline(chirp, dimensions=1)
        adjusted_data = snn_dft_cfar.dft.adjust_1dfft_data(dft_1d)
    fig = snn_dft_cfar.utils.plot_tools.plot_1dfft(adjusted_data, title=title)
    fig.savefig("results/dft1D_scenario1_{}.eps".format(method), dpi=150)
    return

def dft_2d(data_cube, method="snn", title="Spiking DFT"):
    if method=="numpy":
        adjusted_data = fft_2d(data_cube)
    elif method=="snn":
        dft_2d = snn_dft_cfar.pipeline.pipeline(data_cube, dimensions=2)
        adjusted_data = snn_dft_cfar.dft.adjust_2dfft_data(dft_2d)

    fig = snn_dft_cfar.utils.plot_tools.plot_2dfft(adjusted_data, title=title)
    fig.savefig("results/dft2D_scenario1_{}.eps".format(method), dpi=150)
    return


def main(dims=2, method="snn", title="Spiking DFT",
         filename="../data/BBM/scenario1/samples_ch_1_scenario1.txt"):
    # Only the 900 first samples contain information
    data_cube = snn_dft_cfar.utils.read_data.bbm_get_datacube(filename)[:, :900]
    if dims==1:
        chirp_n = 15
        dft_1d(data_cube[chirp_n], method, title)
    if dims==2:
        dft_2d(data_cube, method, title)

if __name__ == "__main__":
    main()
