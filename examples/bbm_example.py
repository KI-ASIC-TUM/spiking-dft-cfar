#!/usr/bin/env python3
"""
Main description of the module.
"""
# Standard libraries
# Local libraries
import snn_dft_cfar.pipeline
import snn_dft_cfar.dft
import snn_dft_cfar.utils.read_data
import snn_dft_cfar.utils.plot_tools


def example_1d(chirp):
    # Run the main pipeline
    dft_1d = snn_dft_cfar.pipeline.pipeline(chirp, dimensions=1)

    # Prepare and plot the 1D DFT data
    adjusted_data = snn_dft_cfar.dft.adjust_1dfft_data(dft_1d)
    fig = snn_dft_cfar.utils.plot_tools.plot_1dfft(adjusted_data)
    fig.savefig("results/snn_dft1D.eps", dpi=150)
    return

def example_2d(data_cube):
    # Run the main pipeline
    dft_2d = snn_dft_cfar.pipeline.pipeline(data_cube, dimensions=2)

    # Prepare and plot the 2D DFT data
    adjusted_data = snn_dft_cfar.dft.adjust_2dfft_data(dft_2d)
    fig = snn_dft_cfar.utils.plot_tools.plot_2dfft(adjusted_data)
    fig.savefig("results/snn_dft2D.eps", dpi=150)
    return

def main(filename="../data/BBM/samples_ch_1_scenario2.txt", dims=1):
    # Only the 900 first samples contain information
    data_cube = snn_dft_cfar.utils.read_data.bbm_get_datacube(filename)[:, :900]
    if dims == 2:
        example_2d(data_cube)
    else:
        chirp_n = 15
        example_1d(data_cube[chirp_n])


if __name__ == "__main__":
    main()
