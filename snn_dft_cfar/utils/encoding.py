#!/usr/bin/env python3
"""
Module containing encoding algorithms for Spiking Neural Networks
"""
# Standard/3rd party libraries
from abc import ABC, abstractmethod
import numpy as np


class Encoder(ABC):
    """
    Encoder abstract class

    Any encoder to be implemented shall be created as an instance of
    this class
    """
    def __init__(self):
        self.spike_train = np.array([])
        return

    @abstractmethod
    def run(self, data, *args):
        return data

    def __call__(self, data, *args):
        return self.run(data, *args)


class LinearFrequencyEncoder(Encoder):
    def __init__(self, min_frequency, max_frequency, min_value, max_value,
                 time_range, time_step=0.0001, random_init=False):
        super().__init__()
        # Store encoder ranges
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.min_value = min_value
        self.max_value = max_value
        self.time_range = time_range

        # Initialize encoder parameters
        self.scale_factor = 0
        self.random_init = random_init
        self.time_step = time_step
        self.spike_trains = np.array([])
        self.setup_encoder_params()

    def setup_encoder_params(self):
        """
        Obtain the parameters of the encoder

        The generated parameters are the period, initial spike time, and the
        total number of spikes that have to be generated for the specified
        time range.
        """
        input_range = self.max_value - self.min_value
        frequency_range = self.max_frequency - self.min_frequency
        self.scale_factor = frequency_range / input_range

    def get_spike_params(self, value):
        """
        Obtain the spike train parameters for a specific input value

        Returns the tuple (period, init_spike, n_spike)
        """
        # Calculate spiking frequency and period
        freq = self.min_frequency + (value-self.min_value)*self.scale_factor
        period = ((1.0 / freq) / self.time_step).astype(np.int64)
        # Generate the first spike at a random position within the range of the
        # obtained period
        if self.random_init:
            init_spike = np.random.uniform(0, period, np.array(value).shape)
            init_spike = init_spike.astype(np.int64)
        else:
            init_spike = np.zeros_like(value)
        return (period, init_spike)

    def run(self, value):
        # If input is vector, transform it into 1xN array
        if len(value.shape)==1:
            value = value.reshape(1, -1)
        rows = value.shape[0]
        cols = value.shape[1]
        periods, init_spikes = self.get_spike_params(value)
        timesteps = int(self.time_range / self.time_step)
        self.spike_trains = np.zeros((rows, cols, timesteps))
        for row, col in np.ndindex((rows, cols)):
            period = periods[row, col]
            init_spike = init_spikes[row, col]
            self.spike_trains[row, col, init_spike::period] = 1
        return self.spike_trains


class TimeEncoder(Encoder):
    """
    Encodes array or single values into the time domain.

    Higher numbers spike at earlier times. Formula of encoding:

    LaTeX:
    t_i = (t_{max}-t_{min}) cdot 
          (1- frac{x_i - x_{min}}{x_{max}-x_{min}} ) + t_{min}

    """

    def __init__(self, t_max, t_min, x_max, x_min):
        """
        Initialization.

        @param t_max : latest possible spike time
        @param t_min : earliest possible spike time
        @param x_max : upper bound for encoded values
        @param x_max : lower bound for encoded values
        """

        super().__init__()
        # Store encoder parameteres
        self.t_max = t_max
        self.t_min = t_min
        self.x_max = x_max
        self.x_min = x_min

    def get_parameters(self):
        """
        Returns the used parameters: t_max, t_min, x_max, x_min.
        """
        return self.t_max, self.t_min, self.x_max, self.x_min

    def set_parameters(self, t_max, t_min, x_max, x_min):
        """
        Set the parameters of TimeEncoder.

        @param t_max : latest possible spike time
        @param t_min : earliest possible spike time
        @param x_max : upper bound for encoded values
        @param x_max : lower bound for encoded values        
        """
        self.t_max = t_max
        self.t_min = t_min
        self.x_max = x_max
        self.x_min = x_min

    def run(self, values):
        """
        Returns the time encoding of the value(s)

        Encoding formula in LaTeX:
        t_i = (t_{max}-t_{min}) cdot 
              (1- frac{x_i - x_{min}}{x_{max}-x_{min}} ) + t_{min}

        @param values: np.array / float / double to encode
        """
        time_range = self.t_max - self.t_min
        value_range = self.x_max - self.x_min
        result = self.t_min + time_range * (1-(values-self.x_min)) / value_range
        return result
