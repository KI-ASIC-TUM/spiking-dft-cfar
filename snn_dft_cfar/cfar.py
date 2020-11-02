#!/usr/bin/env python3
"""
Library containing CFAR implementations
"""# Standard/3rd party libraries
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
# Local libraries


class TraditionalCFAR(ABC):
    """
    Implementation of the traditional CFAR algorithm. 
    """

    def __init__(self,scale_factor,guarding_cells,neighbour_cells):
        """
        Initialization.

        @param scale_factor: how to scale test_cell before compring with 
                             statistical measure
        @param guarding_cells: number of guarding cells (counted from left to 
                               center)
        @param neighbour_cells: number of neighbour cells (as above)
        """

        super().__init__()
        # Store encoder parameteres
        self.scale_factor = scale_factor
        self.guarding_cells = guarding_cells
        self.neighbour_cells = neighbour_cells

        # Initialize arrays for input and output
        self.input_array = np.empty(1)
        self.results = np.empty(1)
        self.threshold = np.empty(1)
        
    def __call__(self, data, *args):
        """
        Calling the object executes the function 'run'.
        """
        return self.run(data, *args)
        
    @abstractmethod
    def statistical_measure(self,np_array):
        """
        Computes the statistical measure for the CFAR algorithm.
        CACFAR: average; OSCFAR: k-th largest.
        """
        return .5

    def compare(self,test_value,stat):
        """
        Compares the test_value against the statistical measure. Takes the 
        scale_factor into consideration.

        @param test_value: value under consideration in CFAR 
        @param stat: statistical measuer computed from neighbours
        """
        return np.heaviside(self.scale_factor*test_value-stat,0)

    def roi_1d(self,np_array):
        """
        roi_1d determines the Region Of Interest, e.g. the test cell and the 
        neighbor values.

        @param np_array: np.array with ndim == 1
        """
        neighbours = np.empty(2*self.neighbour_cells)
        neighbours[:self.neighbour_cells] = np_array[:self.neighbour_cells]
        neighbours[self.neighbour_cells:] = np_array[self.neighbour_cells+
                                             2*self.guarding_cells + 1:]
        test_value = np_array[self.neighbour_cells+self.guarding_cells]
        return test_value, neighbours

    
    def cfar_1d(self,np_array):
        """
        Applies the CFAR algorithm to a 1D array with a sliding window approach.

        @param np_array: np.array with ndim == 1
        """

        # compute sizes related to sliding window 
        N = np_array.size
        window_size = 2*self.neighbour_cells + 2*self.guarding_cells + 1
        total_windows = N-window_size+1
        
        # initialize solution arrays
        self.results = np.zeros(total_windows)
        self.threshold = np.zeros_like(self.results)

        # DEBUG
        #print('input array')
        #print(np_array)

        # iterate in sliding window fashion over input array
        for i in range(total_windows):
            # determine regions of interest
            test_value, neighbour_values = self.roi_1d(
                                            np_array[i:i+window_size])
            # compute statistical measure
            stat = self.statistical_measure(neighbour_values)
            # save threshold
            self.threshold[i] = stat / self.scale_factor
            # save results
            self.results[i] = self.compare(test_value,stat)

            # DEBUG
            #print()
            #print('step')
            #print(i)
            #print('result')
            #print(self.results[i])
            #print('input')
            #print(np_array[i:i+window_size])
            #print('neighbours')
            #print(neighbour_values)
            #print('test value')
            #print(test_value)
            #print('scaled test value')
            #print(self.scale_factor*test_value)
            #print('statistical value')
            #print(stat)

        # return results array    
        return self.results
      
    def cfar_2d(self,np_array):
        """
        Applies the CFAR algorithm to a 2D array with a sliding window approach.

        @param np_array: np.array with ndim == 2
        """

        error = """
        2D CFAR is not yet implemented
        """
        raise SystemError(error)
        
    def run(self, np_array):
        """
        Executes the CFAR algorithm on a given one- or two-dimensional array. 
        Function is executed when object is called.

        @param np_array: np.array with ndim == (1 or 2)
        """
        self.input_array = np_array.copy()
        self.results = np.empty(1)

        dim = np_array.ndim
        if dim == 1:
            return self.cfar_1d(np_array)
        elif dim == 2:
            return self.cfar_2d(np_array)
        else:
            error = """
            CFAR alorithm received wrong a np.array of unexpected dimension {}.
            Expected dimension to be 1 or 2.
            """
            raise SystemError(error)

    def plot(self):
        """
        Visualize the input and output data.
        """
        if self.input_array.ndim == 1:
            self.plot_1d()
        elif self.input_array.ndim == 2:
            error = """
            2D CFAR plotting is not yet implemented
            """
    
    def plot_1d(self):
        """
        Visualize the 1D input and output data.
        """

        # plot input data
        plt.plot(self.input_array,label = 'signal')

        # plot threshold line
        low = self.guarding_cells+self.neighbour_cells
        high = self.guarding_cells+self.neighbour_cells+self.threshold.size
        plt.plot(range(low,high),self.threshold,ls='dotted',lw=1, c='C3', 
                 label='threshold')

        # plot detected peaks
        cntr = 0
        for x in np.where(self.results>=1)[0]:
            if cntr == 0:
                plt.axvline(x+self.guarding_cells+self.neighbour_cells,ls='--',
                            c='C1',lw=1,label='detected peaks')
                cntr +=1
            else:
                plt.axvline(x+self.guarding_cells+self.neighbour_cells,ls='--',
                            c='C1',lw=1)

        # plot boundaries where algorithm works properly
        plt.axvline(self.guarding_cells+self.neighbour_cells-1,ls='--',
                    c='C2',lw=1,label='algorithm boundaries')
        plt.axvline(self.results.size+self.guarding_cells+self.neighbour_cells,
                    ls='--', c='C2',lw=1)
        
        # show plot
        plt.legend()
        plt.grid(True)
        plt.ylabel('signal')
        plt.show()


class CACFAR(TraditionalCFAR):
    """
    Cell Averaging CFAR algorithms.
    """
    def __init__(self, scale_factor,guarding_cells,neighbour_cells):
        """
        Initialization.

        @param scale_factor: how to scale test_cell before compring with 
                             statistical measure
        @param guarding_cells: number of guarding cells (counted from left to 
                               center)
        @param neighbour_cells: number of neighbour cells (as above)
        """
        super(CACFAR,self).__init__(scale_factor,guarding_cells,neighbour_cells)

    def statistical_measure(self,np_array):
        return np.average(np_array)


class OSCFAR(TraditionalCFAR):
    """
    Ordered Statistics CFAR algorithms.
    """
    def __init__(self, scale_factor,guarding_cells,neighbour_cells,k):
        """
        Initialization.

        @param scale_factor: how to scale test_cell before compring with 
                             statistical measure
        @param guarding_cells: number of guarding cells (counted from left to 
                               center)
        @param neighbour_cells: number of neighbour cells (as above)
        @param k: k-th largest value to find for statistical measure (int)
        """
        super(OSCFAR,self).__init__(scale_factor,guarding_cells,neighbour_cells)
        # OSCFAR needs an integer k for determining the k-th largest value
        self.k = k

    def statistical_measure(self,np_array):
        return np.partition(np_array,-self.k)[-self.k]
        