#!/usr/bin/env python3
"""
Library containing CFAR implementations
"""
# Standard/3rd party libraries
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import brian2
# Local libraries
from snn_dft_cfar.utils.encoding import TimeEncoder



class TraditionalCFAR():
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

        self.name = 'generic cfar'

        # Store encoder parameteres
        self.scale_factor = scale_factor
        self.guarding_cells = guarding_cells
        self.neighbour_cells = neighbour_cells

        # Initialize arrays for input and output
        self.input_array = np.empty(1)
        self.results = np.empty(1)
        self.threshold = np.empty(1)
        
        # show threshold
        self.show_threshold = True

    def __call__(self, data, *args):
        """
        Calling the object executes the function 'run'.
        """
        return self.run(data, *args)
        
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
            #stat = self.statistical_measure(neighbour_values)
            # save threshold
            #self.threshold[i] = stat / self.scale_factor
            # save results
            #self.results[i] = self.compare(test_value,stat)
            self.cfar_1d_core(i,test_value,neighbour_values)

        # return results array    
        return self.results

    def cfar_1d_core(self,i,test_value,neighbour_values):
        """
        Computes the statistical measure, threshold, and result.
        @param i: which position of the result is considered
        @param test_value: test value
        @param neighbour_values: values of all neighbours
        """
        # compute statistical measure
        stat = self.statistical_measure(neighbour_values)
        # save threshold
        self.threshold[i] = stat / self.scale_factor
        # save results
        self.results[i] = self.compare(test_value,stat)
      
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
            raise ValueError(error)
    
    def plot_1d(self):
        """
        Visualize the 1D input and output data.
        """

        # plot input data
        plt.plot(self.input_array,label = 'signal')

        # plot threshold line
        if self.show_threshold:
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
        plt.title(self.name)
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
        self.name = 'CA-CFAR'

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
        self.name = 'OS-CFAR'

    def statistical_measure(self,np_array):
        return np.partition(np_array,-self.k)[-self.k]
        

class OSCFAR_SNN(TraditionalCFAR):
    """
    Ordered Statistics CFAR algorithms.
    """
    def __init__(self, scale_factor,guarding_cells,neighbour_cells,
                 k, t_max, t_min, x_max, x_min):
        """
        Initialization.

        @param scale_factor: how to scale test_cell before compring with 
                             statistical measure
        @param guarding_cells: number of guarding cells (counted from left to 
                               center)
        @param neighbour_cells: number of neighbour cells (as above)
        @param k: k-th largest value to find for statistical measure (int)
        @param t_max: largest spike time for simulation
        @param t_min: smallest spike time for simulation
        @param x_max: upper bound for signal values
        @param x_min: lower bound for signal values
        """
        super(OSCFAR_SNN,self).__init__(scale_factor,guarding_cells,
                                        neighbour_cells)
        self.name = 'OS-CFAR SNN'

        # OSCFAR needs an integer k for determining the k-th largest value
        self.k = k

        # SNN does not compute a threshold
        self.show_threshold = False

        # SNN uses time encoding
        self.t_max = t_max
        self.t_min = t_min
        self.x_max = x_max
        self.x_min = x_min

    def cfar_1d_core(self,i,test_value,neighbour_values):
        """
        We replace the cfar_1d_core of the parent class by a functionally 
        equivalent SNN. See notebook for details.

        @param i: which position of the result is considered
        @param test_value: test value
        @param neighbour_values: values of all neighbours
        """

        # initialize time encoder
        time_encoder = TimeEncoder(t_max=self.t_max,t_min=self.t_min,
                                   x_max=self.x_max,x_min=self.x_min)

        # input spike times
        spike_times = np.zeros(neighbour_values.size+1)
        spike_times[:neighbour_values.size] = neighbour_values[:]
        spike_times[-1] = test_value*self.scale_factor
        spike_times=time_encoder(spike_times)
        spike_times = [x*brian2.ms for x in spike_times]

        # define weights
        weights = np.ones(neighbour_values.size+1)
        weights *= -1
        weights[-1] = self.k

        # start Brian2 SNN simulation
        brian2.start_scope()
        simulation_time = 1.1*self.t_max*brian2.ms

        # define neuron group that spikes at specific input times
        input_neurons = brian2.NeuronGroup(len(spike_times), 
                                           'tspike:second',
                                           threshold='t>tspike', 
                                           refractory= simulation_time)
        input_neurons.tspike = spike_times

        # define IF neuron according to previous explanations 
        tau = 10*brian2.ms
        eqs = '''
        dv/dt = 0./tau  : 1
        '''
        compute_neuron = brian2.NeuronGroup(1, eqs, threshold='v>=1', 
                                            refractory= simulation_time, #refractory not working
                                            reset='v = 0', method ='euler')

        # establish connectivity of the network, weights from above
        S = brian2.Synapses(input_neurons,compute_neuron,'w:1',
                            on_pre='v_post += w')
        S.connect()
        S.w = weights

        # monitor spiking behaviour 
        spikemon = brian2.SpikeMonitor(compute_neuron)
        
        # run Brian2 simulation
        brian2.run(simulation_time)
        
        # return 1 if spike occurs, 0 if not.
        self.results[i] = len(spikemon.t)




        

