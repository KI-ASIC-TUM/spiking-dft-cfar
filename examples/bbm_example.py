#!/usr/bin/env python3
"""
Main description of the module.
"""
# Standard libraries
# Local libraries
import snn_dft_cfar.pipeline
import snn_dft_cfar.utils.read_data


def main(filename="../data/BBM/samples_ch_1_scenario2.txt"):
    # Only the 900 first samples contain information
    data_cube = snn_dft_cfar.utils.read_data.bbm_get_datacube(filename)[:, :900]
    snn_dft_cfar.pipeline.pipeline()


if __name__ == "__main__":
    main()
