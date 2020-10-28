#!/usr/bin/env python3
"""
Main description of the module.
"""
# Standard libraries
# Local libraries
from snn_dft_cfar import cfar
from snn_dft_cfar import dft
from snn_dft_cfar.utils import raw_operations
from snn_dft_cfar.utils import encoding


def pipeline(data_cube, single_chirp=False):
    n_chirps, n_samples = data_cube.shape

    # SNN simulation parameters
    time_step = 0.005

    # Normalize all samples between 0 and 1, based on global max and min values
    normalized_cube = raw_operations.normalize(data_cube)
    # Encode the voltage to spikes using rate encoding
    encoder = encoding.LinearFrequencyEncoder(0.1, 100, 0, 1, 5,
                                                         time_step,
                                                         random_init=True)
    encoded_cube = encoder(normalized_cube)
    # Instantiate the DFT SNN class
    snn = dft.FourierTransformSpikingNetwork(
            n_samples, n_chirps, time_step
    )
    if single_chirp:
        layers=1
    else:
        layers=2
    output = snn.run(encoded_cube, layers)
    return output
