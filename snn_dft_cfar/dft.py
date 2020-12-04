#!/usr/bin/env python3
"""
Main description of the module.
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
# Local libraries
import snn_dft_cfar.spiking_dft
from snn_dft_cfar.utils import raw_operations
from snn_dft_cfar.utils import encoding


def dft(raw_data, dimensions, method):
    """
    Return the DFT of the provided data

    @param raw_data: Raw radar sensor data
    @param dimensions: number of dimensions of the input data
    @param method: "snn" or "numpy"
    """
    if method=="SNN":
        result = spiking_dft(raw_data, dimensions)
    elif method=="numpy":
        result = standard_dft(raw_data, dimensions)
    return result


def standard_dft(raw_data, dimensions):
    """
    Perform the standard FFT using NumPy
    """
    if dimensions==1:
        dft_np = np.abs(np.fft.fft(raw_data))
        result = dft_np[1:int(dft_np.size/2)]
    elif dimensions==2:
        dft_np = np.abs(np.fft.fft2(raw_data))
        width, height = dft_np.shape
        # Remove negative values from the range spectrum
        positive_range = dft_np[:, 0:int(height/2)]
        # Re-adjust the plot so the velocity spectrum is centered around zero
        centered = np.vstack(
            (positive_range[int(width/2):, :], positive_range[:int(width/2), :])
        )
        # Place the speed on the horizontal axis, and range in the vertical one
        result = np.flip(np.transpose(centered), axis=0)
    return result


def spiking_dft(raw_data, dimensions, time_step=0.005, adjust=True):
    """
    Returns the output of the S-DFT for the given input

    @param raw_data: np.array containing the radar sensor raw data
    @param dimensions: number of dimensions of the DFT
    """
    if dimensions==1:
        n_samples = raw_data.size
        n_chirps = 1
    elif dimensions==2:
        n_chirps, n_samples = raw_data.shape

    encoded_cube = linear_rate_encoding(raw_data, time_step)
    # Instantiate the DFT SNN class
    snn = snn_dft_cfar.spiking_dft.FourierTransformSpikingNetwork(
        n_samples, n_chirps, time_step
    )

    output = snn.run(encoded_cube, dimensions)
    if adjust:
        output = adjust_snn_dft(output, dimensions)
    return output


def linear_rate_encoding(raw_data, time_step):
    """
    Normalize and encode input data using the LinearFrequencyEncoder
    """
    # Normalize all samples between 0 and 1, based on global max and min values
    normalized_cube = raw_operations.normalize(raw_data)
    # Encode the voltage to spikes using rate encoding
    encoder = encoding.LinearFrequencyEncoder(0.1, 100, 0, 1, 5, time_step,
                                              random_init=True)
    encoded_cube = encoder(normalized_cube)
    return encoded_cube


def adjust_snn_dft(dft_data, dimensions):
    """
    Turn the Spike output into a real-valued map
    """
    spike_sum = dft_data.sum(axis=0)
    n_samples = int(spike_sum.shape[0] / 2)
    n_chirps = int(spike_sum.shape[1] / 4)

    real, imag = get_complex_comps(spike_sum, dimensions, n_samples, n_chirps)
    # Calculate the modulus of the complex valued results, and add a small
    # number for avoiding divide-by-zero errors when applying the logarithm
    modulus = np.sqrt(real**2+imag**2) + 0.1

    if dimensions==1:
        result = modulus[1:int(n_samples/2)]
    else:
        # Remove negative side of the spectrum. Resulting spectrum is "upside-down",
        # so samples have to be taken backwards
        positive_range = modulus[int(n_samples/2):1:-1, :]
        # Re-adjust the plot so the velocity spectrum is centered around zero
        result = np.hstack(
            (positive_range[:, int(n_chirps/2):],
             positive_range[:, :int(n_chirps/2)])
        )
    return result


def get_complex_comps(spike_sum, dimensions, n_samples, n_chirps=1):
    """
    Calculate the real and imaginary components of each bin
    """
    if dimensions==1:
        real_total = spike_sum[:n_samples, 0] - spike_sum[:n_samples, 1]
        imag_total = spike_sum[n_samples:, 0] - spike_sum[n_samples:, 1]

    if dimensions==2:
        real = spike_sum[:, :n_chirps*2]
        imag = spike_sum[:, n_chirps*2:]

        real_total = (
            real[:n_samples, :n_chirps] + real[n_samples:, n_chirps:]
            - (real[n_samples:, :n_chirps] + real[:n_samples, n_chirps:])
        )
        imag_total = (
            imag[:n_samples, :n_chirps] + imag[n_samples:, n_chirps:]
            - (imag[n_samples:, :n_chirps] + imag[:n_samples, n_chirps:])
        )
    return (real_total, imag_total)
