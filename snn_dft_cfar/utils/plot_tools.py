#!/usr/bin/env python3
"""
Function for general plotting function
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np


def format_plotting():
    plt.rcParams['figure.figsize'] = (10, 4)
    plt.rcParams['font.size'] = 22
    #    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = 0.9 * plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = 0.6 * plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = 0.6 * plt.rcParams['font.size']
    # plt.rcParams['savefig.dpi'] = 1000
    plt.rcParams['savefig.format'] = 'eps'
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.loc'] = 'upper right'
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.markersize'] = 3

    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    return

def plot_1dfft(dft_data):
    # Radar parameters
    c = 3 * 10**2   # [m/us]
    f_max = 24 / 2  # [GHz]
    S = 6.55        # [GHz/us]
    # Calculate maximum range
    d_max = (f_max*c) / (2*S)
    
    freq_bins = np.arange(0, d_max, d_max/dft_data.size)
    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(freq_bins, dft_data)
    ax.set_xlabel("Range (m)")
    ax.set_title("Spiking Neural Network DFT")
    plt.show()
    return fig

def plot_2dfft(dft_data):
    # Radar parameters
    f_s = 77        # [GHz]
    c = 3 * 10**2   # [m/us]
    f_max = 24 / 2  # [GHz]
    S = 6.55        # [GHz/us]
    T_chirp = 54    # [us]
    n_chirps = 128
    # Calculate maximum range and velocity of the DFT spectrum
    wavelength = c*1000 / f_s
    d_max = (f_max*c) / (2*S)
    v_max = wavelength / (4*T_chirp)
    v_min = v_max / n_chirps

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(20*np.log10(dft_data), extent=[-v_max, v_max-2*v_min, 0, d_max],
              origin='lower')
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Range (m)")
    ax.set_aspect("auto")
    ax.set_title("Spiking Neural Network")
    plt.show()
    return fig


def plot_cfar(cfar_object):
    """
    Visualize the input and output data.
    """
    if cfar_object.input_array.ndim == 1:
        plot_cfar_1d(cfar_object)
    elif cfar_object.input_array.ndim == 2:
        plot_cfar_2d(cfar_object)

def plot_cfar_1d(cfar_object):
    """
    Visualize the 1D input and output data.
    """

    # plot input data
    plt.plot(cfar_object.input_array,label = 'signal')

    # plot threshold line
    if cfar_object.show_threshold:
        low = cfar_object.guarding_cells+cfar_object.neighbour_cells
        high = cfar_object.guarding_cells+cfar_object.neighbour_cells+ \
                cfar_object.threshold.size
        plt.plot(range(low,high),cfar_object.threshold,ls='dotted',lw=1, c='C3', 
                label='threshold')

    # plot detected peaks
    cntr = 0
    for x in np.where(cfar_object.results>=1)[0]:
        if cntr == 0:
            plt.axvline(x+cfar_object.guarding_cells+ \
                        cfar_object.neighbour_cells,ls='--',
                        c='C1',lw=1,label='detected peaks')
            cntr +=1
        else:
            plt.axvline(x+cfar_object.guarding_cells+ \
                        cfar_object.neighbour_cells,ls='--',
                        c='C1',lw=1)

    # plot boundaries where algorithm works properly
    plt.axvline(cfar_object.guarding_cells+cfar_object.neighbour_cells-1,
                ls='--',c='C2',lw=1,label='algorithm boundaries')
    plt.axvline(cfar_object.results.size+cfar_object.guarding_cells+
                cfar_object.neighbour_cells,
                ls='--', c='C2',lw=1)
    
    # show plot
    plt.legend()
    plt.grid(True)
    plt.ylabel('signal')
    plt.title(cfar_object.name)
    plt.show()

def plot_cfar_2d(cfar_object):
    """
    Visualize the 2D input and output data.
    """

    result = np.zeros_like(cfar_object.input_array)
    temp = cfar_object.guarding_cells+cfar_object.neighbour_cells
    result[temp:temp+cfar_object.results.shape[0],
            temp:temp+cfar_object.results.shape[1]] = cfar_object.results

    fig, (ax1, ax2, ax3) = plt.subplots( 3)
    fig.suptitle('2D {}'.format(cfar_object.name))
    ax1.imshow(cfar_object.input_array)
    ax1.set_xlabel('raw data')
    ax2.imshow(np.log10(cfar_object.input_array))
    ax2.set_xlabel('np log10 data')
    ax3.imshow(result)
    ax3.set_xlabel('cfar detection')
    plt.show()