#!/usr/bin/env python3
"""
Module one-line definition
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
# Local libraries
import snn_dft_cfar.dft
import snn_dft_cfar.utils.read_data


def main():
    """
    Main routine
    """
    params = {
        "min_frequency": 0.05,
        "max_frequency": 50,
        "min_value": 0,
        "max_value": 2,
        "time_range": 40,
        "time_step": 0.0002
    }
    filename = "../data/BBM/scenario4/samples_ch_3_scenario4.txt"
    data_cube = snn_dft_cfar.utils.read_data.bbm_get_datacube(
        filename)[:, :900]

    raw_data = data_cube[15]
    encoded = snn_dft_cfar.dft.linear_rate_encoding(raw_data,
                                                    params).sum(axis=2)

    fig, ax1 = plt.subplots()
    ax1.set_ylabel('normalized voltage')
    ax1.set_xlabel('sample N')
    ax1.plot(raw_data)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Num. spikes")
    ax2.plot(encoded[0, :], color="tab:red")
    plt.show()


if __name__ == "__main__":
    main()
