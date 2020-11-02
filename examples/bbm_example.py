#!/usr/bin/env python3
"""
Main description of the module.
"""
# Standard libraries
# Local libraries
import snn_dft_cfar.pipeline
import snn_dft_cfar.dft
import snn_dft_cfar.utils.read_data
import snn_dft_cfar.utils.plot_tools
from snn_dft_cfar.cfar import CACFAR, OSCFAR, OSCFAR_SNN


def example_1d(chirp, CFAR, guarding_cells, neighbour_cells, 
                   scale_factor, k, t_max, t_min, x_max, x_min):
    """
    Run DFT and CFAR on a one dimensional numpy array. All parameters are 
    documented in the CFAR child classes. Besides:

    @param chirp: 1D numpy array that corresponds to the signal.
    """
    # Run the main pipeline
    dft_1d = snn_dft_cfar.pipeline.pipeline(chirp, dimensions=1)

    # Prepare and plot the 1D DFT data
    adjusted_data = snn_dft_cfar.dft.adjust_1dfft_data(dft_1d)

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

    # visualize result
    cfar.plot()

def example_2d(data_cube):
    """
    Run DFT and CFAR on a two dimensional numpy array. CFAR 2D not implemented 
    yet.

    @param data_cube: 2D numpy array that corresponds to the signals.
    """
    # Run the main pipeline
    dft_2d = snn_dft_cfar.pipeline.pipeline(data_cube, dimensions=2)

    # Prepare and plot the 2D DFT data
    adjusted_data = snn_dft_cfar.dft.adjust_2dfft_data(dft_2d)
    fig = snn_dft_cfar.utils.plot_tools.plot_2dfft(adjusted_data)
    fig.savefig("results/snn_dft2D.eps", dpi=150)
    return

def main(filename="../data/BBM/samples_ch_1_scenario2.txt", dims=1):

    # Load data cube from simulation data;
    # Only the 900 first samples contain information
    data_cube = snn_dft_cfar.utils.read_data.bbm_get_datacube(filename)[:, :900]

    # Define CFAR parameters (parameteres are chosen such that they treat the 
    # sample well). 
    guarding_cells = 6
    neighbour_cells = 15
    scale_factor = 0.2
    k = 6

    # Parameters for time encoding, x_max / x_min must bound DFT values
    t_max, t_min, x_max, x_min = 50, 0, 100, 0

    # Choose CFAR implentation. Available: 'CACFAR', 'OSCFAR', 'OSCFARSNN'
    # The latter two behave equivalently but 'OSCFAR' is significantly faster. 
    # Use it for prototyping.
    CFAR = 'OSCFARSNN'
    
    # Run pipeline (DFT+CFAR); currently 2D CFAR is not implented
    if dims == 2:
        example_2d(data_cube)
    else:
        chirp_n = 15
        example_1d(data_cube[chirp_n],CFAR, guarding_cells, neighbour_cells, 
                   scale_factor, k, t_max, t_min, x_max, x_min)


if __name__ == "__main__":
    main()
