#!/usr/bin/env python3
"""
Library containing 2D-DFT implementations
"""
# Standard libraries
import numpy as np
# Local libraries


class FourierTransformArtificialNetwork():
    def __init__(self,
                 n_input,
                 n_chirps,
                 time_step=0.001,
                 total_time=5,
                 normalize=True):
        self.n_input = n_input
        self.n_chirps = n_chirps
        self.normalize = normalize

    def calculate_weights(self):
        self.weights = []
        self.__calculate_weights_layer1()
        self.__calculate_weights_layer2()

    def __calculate_weights_layer1(self):
        """
        Calculate 1-D FFT coefficients based on algorithm
        """
        # Constant present in all weight elements
        c_1 = 2 * np.pi / self.n_input

        # Calculate the content of the cosines/sines as a dot product
        n = np.arange(self.n_input).reshape(self.n_input, 1)
        k1 = np.arange(self.n_input).reshape(1, self.n_input)
        trigonometric_factors = np.dot(n, k1) * c_1

        # Create the weights matrix for a single chirp
        real_weights = np.cos(trigonometric_factors)
        imag_weights = -np.sin(trigonometric_factors)

        # Normalize the weights by dividing by the input length. This is due to
        # the properties of the FFT
        if self.normalize:
            real_weights /= self.n_input
            imag_weights /= self.n_input

        # Append a negative version of the weights, so negative values can be
        # calculated with ReLU-like layers
        real_weights = np.vstack((real_weights, -real_weights))
        imag_weights = np.vstack((imag_weights, -imag_weights))

        # Create the final weights matrix by stacking the same matrix M times
        # where M is the number of chirps
        self.weights.append((real_weights, imag_weights))
        return self.weights

    def __calculate_weights_layer2(self):
        """
        Calculate 2-D FFT coefficients based on algorithm
        """
        # Constant present in all weight elements
        c_1 = 2 * np.pi / self.n_chirps

        # Calculate the content of the cosines/sines as a dot product
        m = np.arange(self.n_chirps).reshape(self.n_chirps, 1)
        k2 = np.arange(self.n_chirps).reshape(1, self.n_chirps)
        trigonometric_factors = np.dot(m, k2) * c_1

        # Create the weights matrix for a single chirp
        cosine_factors = np.cos(trigonometric_factors)
        sine_factors = np.sin(trigonometric_factors)

        real_weights = np.hstack((cosine_factors, sine_factors))
        imag_weights = np.hstack((-sine_factors, cosine_factors))

        # Normalize the weights by dividing by the chirp length.
        if self.normalize:
            real_weights /= self.n_chirps
            imag_weights /= self.n_chirps

        # Append a negative version of the weights, so negative values can be
        # calculated with ReLU-like layers
        real_weights = np.vstack((real_weights, -real_weights))
        imag_weights = np.vstack((imag_weights, -imag_weights))

        # Create the final weights matrix by stacking the same matrix M times
        # where M is the number of chirps
        self.weights.append((real_weights, imag_weights))
        return self.weights

    def run(self, input_values, layers=2):
        """
        Main routine for implementing the SNN
        """
        self.calculate_weights()
        ## Layer 1
        z_1_real = np.dot(self.weights[0][0], input_values.transpose())
        z_1_imag = np.dot(self.weights[0][1], input_values.transpose())
        z_1 = np.hstack((z_1_real, z_1_imag))
        # ReLU functionality
        z_1 = np.where(z_1 > 0, z_1, 0)
        if layers == 1:
            return z_1

        ## Layer 2
        z_2_real = np.dot(self.weights[1][0], z_1.transpose())
        z_2_imag = np.dot(self.weights[1][1], z_1.transpose())
        z_2 = np.hstack((z_2_real, z_2_imag))
        # ReLU functionality
        z_2 = np.where(z_1 > 0, z_1, 0)
        return self.spike_trains_l2
