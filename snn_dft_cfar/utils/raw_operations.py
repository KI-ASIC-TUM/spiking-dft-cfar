#!/usr/bin/env python3
"""
Module containing auxiliary operations applied to raw data
"""
# Standard libraries
import numpy as np
# Local libraries


def get_rms(voltage, sample_time):
    """
    Calculate the root-mean-square value of an input voltage sequence
    """
    voltage_arr = np.array(voltage)
    total_time = sample_time * voltage_arr.size
    squared_voltage = voltage_arr**2
    rms = np.sqrt((squared_voltage * sample_time).sum() / total_time)
    return rms


def normalize(input_data):
    """
    Convert the input data to the range [0..1]
    """
    min_value = input_data.min()
    # Remove bias from the input signal
    no_offset = input_data - min_value
    # Normalize to [0..1]
    max_value = no_offset.max()
    normalized = no_offset / max_value
    return normalized
