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


def main(filename="../data/BBM/samples_ch_1_scenario2.txt"):
    # Only the 900 first samples contain information
    data_cube = snn_dft_cfar.utils.read_data.bbm_get_datacube(filename)[:, :900]
    # Run the main pipeline
    dft_2d = snn_dft_cfar.pipeline.pipeline(data_cube)

    # Prepare and plot the 2D DFT data
    adjusted_data = snn_dft_cfar.dft.adjust_2dfft_data(dft_2d)
    fig = snn_dft_cfar.utils.plot_tools.plot_2dfft(adjusted_data)
    fig.savefig("results/snn_dft2D.eps", dpi=150)


if __name__ == "__main__":
    main()
