#!/usr/bin/env python3
"""
Main description of the module.
"""
# Standard libraries
import numpy as np
from timeit import default_timer as timer

# Local libraries
import snn_dft_cfar.pipeline
import snn_dft_cfar.dft
import snn_dft_cfar.utils.read_data
import snn_dft_cfar.utils.plot_tools
from snn_dft_cfar.utils.plot_tools import plot_cfar
from snn_dft_cfar.cfar import CACFAR, OSCFAR, OSCFAR_SNN


def print_benchmark_message(fft,fft_to_cfar,cfar_time,cfar_time_per_sample,
                            brian_time, brian_time_per_sample):
    print()
    print(24*'=','Benchmark SNN Radar Processing',24*'=')
    print('FFT                     : {0:.6f}'.format(fft))
    print('FFT to CFAR             : {0:.6f}'.format(fft_to_cfar))
    print('CFAR                    : {0:.6f}'.format(cfar_time))
    print('CFAR per iteration      : {0:.6f}'.format(cfar_time_per_sample))
    print('Brian 2 in CFAR         : {0:.6f}'.format(brian_time))
    print('Brian 2 in CFAR per it. : {0:.6f}'.format(brian_time_per_sample))
    print(80*'=','\n')

def example_1d(chirp, FT, CFAR, guarding_cells, neighbour_cells, 
                   scale_factor, k, t_max, t_min, x_max, x_min):
    """
    Run DFT and CFAR on a one dimensional numpy array. All parameters are 
    documented in the CFAR child classes. Besides:

    @param chirp: 1D numpy array that corresponds to the signal.
    """

    
    if FT == 'FFT':
        start = timer()
        adjusted_data = np.abs(np.fft.fft(chirp))
        end = timer()
        fft_time = end-start
        # No processing in FFT mode, hence no time in benchmark
        fft_to_cfar_time = 0
    elif FT == 'DFTSNN':
        # Run the main pipeline
        start = timer()
        dft_1d = snn_dft_cfar.pipeline.pipeline(chirp, dimensions=1)
        end = timer()
        fft_time = end-start
        
        # Prepare and plot the 1D DFT data
        start = timer()
        adjusted_data = snn_dft_cfar.dft.adjust_1dfft_data(dft_1d)
        end = timer()
        fft_to_cfar_time = end-start
    else:
        error = 'Your choice for FT ({}) is not valid.'.format(FT)
        raise ValueError(error) 
    

    # instanciate CFAR dependening on choice
    if CFAR == 'OSCFARSNN':
        cfar = OSCFAR_SNN(scale_factor,guarding_cells,neighbour_cells,k,
                          t_max, t_min, x_max, x_min)
    elif CFAR == 'OSCFAR':
        cfar = OSCFAR(scale_factor,guarding_cells,neighbour_cells,k)
    elif CFAR == 'CACFAR':
        cfar = CACFAR(scale_factor,guarding_cells,neighbour_cells)
    else:
        error = 'Your choice for CFAR ({}) is not valid.'.format(CFAR)
        raise ValueError(error)

    # run CFAR algorithm
    cfar(adjusted_data)

    # print benchmark
    if CFAR == 'CACFAR' or CFAR == 'OSCFAR':
        brian_time = 0
    else:
        brian_time = cfar.brian_sim_time
    print_benchmark_message(fft_time,fft_to_cfar_time,cfar.processing_time,
                            cfar.processing_time/cfar.results.size,
                            brian_time, brian_time /cfar.results.size )

    # visualize result
    plot_cfar(cfar)

def example_2d(data_cube, FT, CFAR, guarding_cells, neighbour_cells, 
                   scale_factor, k, t_max, t_min, x_max, x_min):
    """
    Run DFT and CFAR on a two dimensional numpy array. CFAR 2D not implemented 
    yet.

    @param data_cube: 2D numpy array that corresponds to the signals.
    """
    
    # Run Fourier transformation
    if FT == 'FFT':
        start = timer()
        adjusted_data = np.abs(np.fft.fft2(data_cube))
        end = timer()
        fft_time = end-start
        # No processing in FFT mode, hence no time in benchmark
        fft_to_cfar_time = 0
    elif FT == 'DFTSNN':
        # run DFT SNN
        start = timer()
        dft_2d = snn_dft_cfar.pipeline.pipeline(data_cube, dimensions=2)
        end = timer()
        fft_time = end-start
        # Prepare and plot the 2D DFT data
        start = timer()
        adjusted_data = snn_dft_cfar.dft.adjust_2dfft_data(dft_2d)
        end = timer()
        fft_to_cfar_time = end-start
    else:
        error = 'Your choice for CFAR ({}) is not valid.'.format(CFAR)
        raise ValueError(error)

    # instanciate CFAR dependening on choice
    if CFAR == 'OSCFARSNN':
        cfar = OSCFAR_SNN(scale_factor,guarding_cells,neighbour_cells,k,
                          t_max, t_min, x_max, x_min)
    elif CFAR == 'OSCFAR':
        cfar = OSCFAR(scale_factor,guarding_cells,neighbour_cells,k)
    elif CFAR == 'CACFAR':
        cfar = CACFAR(scale_factor,guarding_cells,neighbour_cells)
    else:
        error = 'Your choice for CFAR ({}) is not valid.'.format(CFAR)
        raise ValueError(error)

    # run CFAR algorithm
    cfar(adjusted_data)

    # print benchmark
    if CFAR == 'CACFAR' or CFAR == 'OSCFAR':
        brian_time = 0
    else:
        brian_time = cfar.brian_sim_time
    print_benchmark_message(fft_time,fft_to_cfar_time,cfar.processing_time,
                            cfar.processing_time/cfar.results.size,
                            brian_time, brian_time /cfar.results.size )

    # visualize result
    plot_cfar(cfar)

def main(filename="../data/BBM/samples_ch_1_scenario2.txt", dims=1):

    # Load data cube from simulation data;
    # Only the 900 first samples contain information
    data_cube = snn_dft_cfar.utils.read_data.bbm_get_datacube(filename)[:, :900]

    # Choose FT implentation. Available: 'FFT', 'DFTSNN'
    # The two behave roughly equivalently but 'FFT' is significantly faster. 
    # Use it for prototyping. Note that 'DFTSNN' contains additional processing 
    # to properly arrange the data.
    FT = 'FFT'

    # Choose CFAR implentation. Available: 'CACFAR', 'OSCFAR', 'OSCFARSNN'
    # The latter two behave equivalently but 'OSCFAR' is significantly faster. 
    # Use it for prototyping.
    CFAR = 'OSCFAR'
    
    # Run pipeline (DFT+CFAR); currently 2D CFAR is not implented
    if dims == 2:
        
        # Define CFAR parameters (parameteres are chosen such that they treat 
        # the sample well). 
        guarding_cells = 3
        neighbour_cells = 4
        scale_factor = 0.2
        k = 8

        # Parameters for time encoding, x_max / x_min must bound DFT values
        t_max, t_min, x_max, x_min = 50, 0, 100, 0

        example_2d(data_cube,FT,CFAR, guarding_cells, neighbour_cells, 
                   scale_factor, k, t_max, t_min, x_max, x_min)
    else:
        chirp_n = 15

        # Define CFAR parameters (parameteres are chosen such that they treat 
        # the sample well). 
        guarding_cells = 6
        neighbour_cells = 15
        scale_factor = 0.2
        k = 6

        # Parameters for time encoding, x_max / x_min must bound DFT values
        t_max, t_min, x_max, x_min = 50, 0, 100, 0

        example_1d(data_cube[chirp_n],FT,CFAR, guarding_cells, neighbour_cells, 
                   scale_factor, k, t_max, t_min, x_max, x_min)


if __name__ == "__main__":
    main(dims=1)
