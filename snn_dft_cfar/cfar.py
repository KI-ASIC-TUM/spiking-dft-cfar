#!/usr/bin/env python3
"""
Library containing CFAR implementations
"""
# Standard/3rd party libraries
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
# Local libraries
from snn_dft_cfar.utils.encoding import TimeEncoder
from snn_dft_cfar.utils.if_neuron import simulate_IF_neuron


class TraditionalCFAR():
    """
    Implementation of the traditional CFAR algorithm. 
    """
    def __init__(self, scale_factor, guarding_cells, neighbour_cells,
                 zero_padding):
        """
        Initialization

        @param scale_factor: how to scale test_cell before compring with 
         statistical measure
        @param guarding_cells: number of guarding cells (counted from
         left to center)
        @param neighbour_cells: number of neighbour cells (as above)
        @param zero_padding: bool indicating whether to compute CFAR at
         the borders by enlarging the frame with zeros
        """
        self.name = 'generic cfar'
        self.processing_time = 0.

        # Store encoder parameters
        self.scale_factor = scale_factor
        self.guarding_cells = guarding_cells
        self.neighbour_cells = neighbour_cells

        # Initialize arrays for input and output
        self.input_array = np.empty(1)
        self.results = np.empty(1)
        self.threshold = np.empty(1)

        # show threshold
        #TODO: Remove variable if unused
        self.show_threshold = True

        # use zero padding
        self.zero_padding = zero_padding

    def __call__(self, data, *args):
        """
        Calling the object executes the function 'run'.
        """
        return self.run(data, *args)

    def statistical_measure(self, np_array):
        """
        Computes the statistical measure for the CFAR algorithm.

        CACFAR: average; OSCFAR: k-th largest.
        """
        raise NotImplementedError("This method is for children classes")

    def compare(self, test_value, stat):
        """
        Compares the test_value against the statistical measure.

        It takes the scale_factor into consideration.

        @param test_value: value under consideration in CFAR 
        @param stat: statistical measuer computed from neighbours
        """
        return np.heaviside(self.scale_factor * test_value - stat, 0)

    def roi_1d(self, np_array):
        """
        Determines the 1D Region Of Interest

        e.g. the test cell and the neighbor values.

        @param np_array: np.array with ndim == 1
        """
        neighbours = np.empty(2*self.neighbour_cells)
        neighbours[:self.neighbour_cells] = np_array[:self.neighbour_cells]
        neighbours[self.neighbour_cells:] = np_array[self.neighbour_cells
                                                     + 2*self.guarding_cells
                                                     +1
                                                     :]
        test_value = np_array[self.neighbour_cells + self.guarding_cells]
        return test_value, neighbours

    def roi_2d(self, np_array):
        """
        Determines the 2D Region Of Interest

        e.g. the test cell and the neighbor values.

        @param np_array: np.array with ndim == 2
        """
        no_neighbours = np_array.size - (2*self.guarding_cells + 1)**2
        neighbours = np.empty(no_neighbours)

        # Read out neighbours
        centre = 0
        for row in range(np_array.shape[0]):
            if row < self.neighbour_cells:
                chunk = np_array[row, :]
                neighbours[centre : centre+chunk.size] = chunk
                centre += chunk.size
            elif row < (1 + self.neighbour_cells + 2*self.guarding_cells):
                chunk = np_array[row, :self.neighbour_cells]
                neighbours[centre : centre+chunk.size] = chunk
                centre += chunk.size
                chunk = np_array[row, self.neighbour_cells +
                                 2*self.guarding_cells + 1 :]
                neighbours[centre : centre+chunk.size] = chunk
                centre += chunk.size
            else:
                chunk = np_array[row, :]
                neighbours[centre : centre + chunk.size] = chunk
                centre += chunk.size

        if (centre != no_neighbours):
            error = """
            In roi_2d. centre ({}) does not add up to all neighbor cells ({}).
            """.format(centre, no_neighbours)
            raise ValueError(error)

        # test value can be found in the center
        test_value = np_array[self.neighbour_cells + self.guarding_cells,
                              self.neighbour_cells + self.guarding_cells]
        return test_value, neighbours

    def cfar_1d(self, np_array):
        """
        Applies the CFAR algorithm to a 1D array with a sliding window

        @param np_array: np.array with ndim == 1
        """
        # compute sizes related to sliding window
        N = np_array.size
        window_size = 2*self.neighbour_cells + 2*self.guarding_cells + 1
        total_windows = N - window_size + 1

        # initialize solution arrays
        self.results = np.zeros(total_windows)
        self.threshold = np.zeros_like(self.results)

        # iterate in sliding window fashion over input array
        for i in range(total_windows):
            # determine regions of interest
            test_value, neighbours = self.roi_1d(np_array[i : i+window_size])
            self.cfar_1d_core(i, test_value, neighbours)

        # return results array
        return self.results

    def cfar_1d_core(self, i, test_value, neighbour_values):
        """
        Computes the statistical measure, threshold, and result

        @param i: which position of the result is considered
        @param test_value: test value
        @param neighbour_values: values of all neighbours
        """
        # compute statistical measure
        stat = self.statistical_measure(neighbour_values)
        # save threshold
        self.threshold[i] = stat / self.scale_factor
        # save results
        self.results[i] = self.compare(test_value, stat)

    def cfar_2d(self, np_array):
        """
        Applies the CFAR algorithm to a 2D array with a sliding window

        @param np_array: np.array with ndim == 2
        """
        # compute sizes related to sliding window
        N_x, N_y = np_array.shape
        window_size = 2*self.neighbour_cells + 2*self.guarding_cells + 1
        total_windows_x = N_x - window_size + 1
        total_windows_y = N_y - window_size + 1

        # initialize solution arrays
        self.results = np.zeros(total_windows_x*total_windows_y)
        self.threshold = np.zeros_like(self.results)

        # iterate in sliding window fashion over input array
        for i, j in np.ndindex(total_windows_x, total_windows_y):
            # determine regions of interest
            test_value, neighbour_values = self.roi_2d(
                    np_array[i : i+window_size, j : j+ window_size])
            self.cfar_1d_core(total_windows_y*i + j,
                              test_value,
                              neighbour_values
                             )

        # reshape results
        self.results = self.results.reshape((total_windows_x, total_windows_y))
        self.threshold = self.threshold.reshape(
                (total_windows_x, total_windows_y))

        # return results array
        return self.results

    def run(self, np_array):
        """
        Executes the CFAR algorithm on a given 1D or 2D array

        Function is executed when object is called.

        @param np_array: np.array with ndim == (1 or 2)
        """
        # apply zero padding to arrays
        dim = np_array.ndim
        #TODO: Send zero padding functionality to isolated method
        if self.zero_padding:
            padding = self.guarding_cells + self.neighbour_cells
            if dim == 1:
                tmp = np.zeros(np_array.size + 2*padding)
                tmp[padding : padding+np_array.size] = np_array
                np_array = tmp
                del (tmp)
            elif dim == 2:
                r_dim, c_dim = np_array.shape
                tmp = np.zeros((r_dim + 2*padding, c_dim + 2*padding))
                tmp[padding : padding+r_dim,
                    padding : padding+c_dim] = np_array
                np_array = tmp
            else:
                error = """
                CFAR alorithm received wrong a np.array of unexpected
                dimension {}.Expected dimension 1 or 2.
                """.format(dim)
                raise ValueError(error)
        else:
            pass

        self.input_array = np_array.copy()
        self.results = np.empty(1)

        # run CFAR algorithm.
        start = timer()
        if dim == 1:
            self.cfar_1d(np_array)
        elif dim == 2:
            self.cfar_2d(np_array)
        end = timer()
        self.processing_time = end - start

        return self.results


class CACFAR(TraditionalCFAR):
    """
    Cell Averaging CFAR algorithm
    """
    def __init__(self,
                 scale_factor,
                 guarding_cells,
                 neighbour_cells,
                 zero_padding
                ):
        """
        Initialization

        @param scale_factor: how to scale test_cell before compring with 
                             statistical measure
        @param guarding_cells: number of guarding cells (counted from left to 
                               center)
        @param neighbour_cells: number of neighbour cells (as above)
        @param zero_padding: bool indicating whether to compute CFAR at
         the borders by enlarging the frame with zeros
        """
        super(CACFAR, self).__init__(scale_factor, guarding_cells,
                                     neighbour_cells, zero_padding)
        self.name = 'CA-CFAR'

    def statistical_measure(self, np_array):
        return np.average(np_array)


class OSCFAR(TraditionalCFAR):
    """
    Ordered Statistics CFAR algorithm
    """
    def __init__(self,
                 scale_factor,
                 guarding_cells,
                 neighbour_cells,
                 k,
                 zero_padding,
                 **kwargs
                ):
        """
        Initialization

        @param scale_factor: how to scale test_cell before compring with 
                             statistical measure
        @param guarding_cells: number of guarding cells (counted from left to 
                               center)
        @param neighbour_cells: number of neighbour cells (as above)
        @param k: k-th largest value to find for statistical measure (int)
        @param zero_padding: bool indicating whether to compute CFAR at
         the borders by enlarging the frame with zeros
        """
        super(OSCFAR, self).__init__(scale_factor, guarding_cells,
                                     neighbour_cells, zero_padding)
        # OSCFAR needs an integer k for determining the k-th largest value
        self.k = k
        self.name = 'OS-CFAR'

    def statistical_measure(self, np_array):
        return np.partition(np_array, -self.k)[-self.k]


class OSCFAR_SNN(TraditionalCFAR):
    """
    Ordered Statistics CFAR algorithms.
    """
    def __init__(self,
                 scale_factor,
                 guarding_cells,
                 neighbour_cells,
                 k,
                 zero_padding,
                 t_max,
                 t_min,
                 x_max,
                 x_min,
                 t_step=0.01
                ):
        """
        Initialization.

        @param scale_factor: how to scale test_cell before compring with 
                             statistical measure
        @param guarding_cells: number of guarding cells (counted from left to 
                               center)
        @param neighbour_cells: number of neighbour cells (as above)
        @param k: k-th largest value to find for statistical measure (int)
        @param zero_padding: bool indicating whether to compute CFAR at
         the borders by enlarging the frame with zeros
        @param t_max: largest spike time for simulation
        @param t_min: smallest spike time for simulation
        @param x_max: upper bound for signal values
        @param x_min: lower bound for signal values
        """
        super(OSCFAR_SNN, self).__init__(scale_factor, guarding_cells,
                                         neighbour_cells, zero_padding)
        self.name = 'OS-CFAR SNN'
        self.sim_time = 0.

        # OSCFAR needs an integer k for determining the k-th largest value
        self.k = k

        # SNN does not compute a threshold
        self.show_threshold = False

        # SNN uses time encoding
        self.t_max = t_max
        self.t_min = t_min
        self.x_max = x_max
        self.x_min = x_min
        self.t_step = t_step

    def cfar_1d_core(self, i, test_value, neighbour_values):
        """
        Replace the parent class core by a functionally equivalent SNN

        See notebook for details.

        @param i: which position of the result is considered
        @param test_value: test value
        @param neighbour_values: values of all neighbours
        """
        # initialize time encoder
        time_encoder = TimeEncoder(t_max=self.t_max,
                                   t_min=self.t_min,
                                   x_max=self.x_max,
                                   x_min=self.x_min
                                  )

        spike_times = np.zeros(neighbour_values.size+1)
        spike_times[:neighbour_values.size] = neighbour_values[:]
        spike_times[-1] = test_value * self.scale_factor
        spike_times = time_encoder(spike_times)

        # define weights
        weights = np.ones(neighbour_values.size + 1)
        weights *= -1
        weights[-1] = self.k

        # simulate IF neuron
        start_local = timer()
        res = simulate_IF_neuron(self.t_min, self.t_max, self.t_step,
                                 spike_times, weights)
        end_local = timer()
        self.sim_time += (end_local - start_local)

        # return 1 if spike occurs, 0 if not.
        self.results[i] = res
