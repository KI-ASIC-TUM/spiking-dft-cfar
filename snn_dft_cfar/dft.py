#!/usr/bin/env python3
"""
Library containing 2D-DFT implementations
"""
# Standard libraries
import numpy as np
# Local libraries

class FourierTransformSpikingNetwork():
    def __init__(self, n_input, n_chirps, time_step=0.001, normalize=True):
        self.n_input = n_input
        self.n_chirps = n_chirps
        self.normalize = normalize

        # Spiking neuron attributes
        self.v_threshold = 10
        self.v_rest = 0
        self.bias = 0
        self.v_membrane = [np.zeros((self.n_input*2, 2*self.n_chirps)),
                           np.zeros((self.n_input*2, 4*self.n_chirps)),
                          ]
        self.v_membrane[0] -= self.v_rest
        self.v_membrane[1] -= self.v_rest
        self.spikes = np.zeros((self.n_input, 2*self.n_chirps))
        self.spike_trains_l1 = np.array([])
        self.spike_trains_l2 = np.array([])

        # SNN simulation parameters
        self.time_step = time_step
        self.sim_time = 0
        self.total_time = 5

    def calculate_weights(self):
        self.weights = []
        self.__calculate_weights_layer1()
        self.__calculate_weights_layer2()

    def __calculate_weights_layer1(self):
        """
        Calculate 1-D FFT coefficients based on algorithm
        """
        # Constant present in all weight elements
        c_1 = 2*np.pi / self.n_input

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
        c_1 = 2*np.pi / self.n_chirps

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

    def update_input_spikes(self, spike_trains):
        """
        Assess ocurrence of spikes happening at current simulation time
        """
        spikes = np.zeros((self.n_chirps, self.n_input))
        # Check if there are spikes between current sim_time and the next
        # simulation step
        for row, col in np.ndindex(spikes.shape):
            spike_train = spike_trains[row, col]
            spike = np.any((spike_train >= self.sim_time)
                            & (spike_train < (self.sim_time+self.time_step))
                          )
            spikes[row, col] = int(spike)
        return spikes

    def update_input_currents(self, input_spikes, weights):
        """
        Calculate the total current that circulates inside each neuron
        """
        z_real = np.dot(weights[0], input_spikes.transpose())
        z_imag = np.dot(weights[1], input_spikes.transpose())
        # Add bias to the result and multiply by threshold voltage
        z_real += self.bias
        z_imag += self.bias
        z_real *= self.v_threshold
        z_imag *= self.v_threshold
        return (z_real, z_imag)

    def generate_spikes(self, z, layer):
        """
        Determine which neurons spike based on membrane potential
        """
        # Calculate the charge of the membrane relative to the threshold
        voltage = self.v_membrane[layer] + z - self.v_threshold
        # Generate a spike when the relative voltage is positive
        self.spikes = np.where(voltage>0, 1, 0)
        return self.spikes

    def update_membrane_potential(self, z, layer):
        """
        Update membrane potential of each neuron

        The membrane potential increases based on the input current, and
        it returns to the rest voltage after a spike
        """
        self.v_membrane[layer] += z - self.spikes*self.v_threshold
        return self.v_membrane

    def run(self, spike_trains, layers=2):
        """
        Main routine for implementing the SNN
        """
        self.calculate_weights()
        sim_size = int(self.total_time / self.time_step)
        self.spike_trains_l1 = np.zeros((sim_size, 2*self.n_input, 2*self.n_chirps))
        self.spike_trains_l2 = np.zeros((sim_size, 2*self.n_input, 4*self.n_chirps))
        # Simulate the SNN until the simulation time reaches the limit
        for idx in range(sim_size):
            ## Layer 1
            # Update presence of input spikes
            input_spikes = spike_trains[:, :, idx]
            # Update input current
            z_re, z_im = self.update_input_currents(input_spikes, self.weights[0])
            z = np.hstack((z_re, z_im))
            # Update spike generation
            self.generate_spikes(z, layer=0)
            self.spike_trains_l1[idx] = self.spikes
            # Update membrane potential
            self.update_membrane_potential(z, layer=0)
            if layers==1:
                continue

            ## Layer 2
            # Update input current
            z_re, z_im = self.update_input_currents(self.spikes, self.weights[1])
            z = np.vstack((z_re, z_im)).transpose()
            # Update spike generation
            self.generate_spikes(z, layer=1)
            self.spike_trains_l2[idx] = self.spikes
            # Update membrane potential
            self.update_membrane_potential(z, layer=1)

            # Increase current simulation time
            self.sim_time += self.time_step
        if layers==1:
            return self.spike_trains_l1
        return self.spike_trains_l2


def adjust_1dfft_data(input_data):
    """
    Adjust data after spiking 2D DFT, so it can be easily plotted
    """
    spike_sum = input_data.sum(axis=0)
    real = spike_sum[:900, 0] - spike_sum[:900, 1]
    imag = spike_sum[900:, 0] - spike_sum[900:, 1]
    modulus = np.sqrt(real**2+imag**2) + 0.1
    result = modulus[1:450]
    return result

def adjust_2dfft_data(input_data):
    """
    Adjust data after spiking 2D DFT, so it can be easily plotted
    """
    # Separate real from imaginary parts,
    # and combine positive and negative quadrants
    spike_sum = input_data.sum(axis=0)
    width, height = spike_sum.shape
    n_samples = int(width/2)
    n_chirps = int(height/4)

    real = spike_sum[:, :n_chirps*2]
    imag = spike_sum[:, n_chirps*2:]

    real_pos = real[:n_samples, :n_chirps]
    real_pos += real[n_samples:, n_chirps:]
    real_neg = real[n_samples:, :n_chirps]
    real_neg += real[:n_samples, n_chirps:]
    real_total = real_pos - real_neg
    imag_pos = imag[:n_samples, :n_chirps]
    imag_pos += imag[n_samples:, n_chirps:]
    imag_neg = imag[n_samples:, :n_chirps]
    imag_neg += imag[:n_samples, n_chirps:]
    imag_total = imag_pos - imag_neg

    # Calculate the modulus of the complex valued results, and add a small
    # number for avoiding divide-by-zero errors when applying the logarithm
    modulus = np.sqrt(real_total**2+imag_total**2) + 0.1

    width, height = modulus.shape
    # Remove negative side of the spectrum. Resulting spectrum is "upside-down",
    # so samples have to be taken backwards
    positive_range = modulus[int(width/2):1:-1, :]
    # Re-adjust the plot so the velocity spectrum is centered around zero
    result = np.copy(positive_range)
    result[:, :int(height/2)] = positive_range[:, int(height/2):]
    result[:, int(height/2):] = positive_range[:, :int(height/2)]

    return result
