#!/usr/bin/env python3
"""
Function for general plotting function
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np


def format_plotting():
    # plt.rcParams['figure.figsize'] = (10, 4)
    plt.rcParams['font.size'] = 20
    #    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1 * plt.rcParams['font.size']
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

def plot_dft(dft_data, title, show=True, ax=None):
    if dft_data.ndim==1:
        fig = plot_1dfft(dft_data, title, show)
    elif dft_data.ndim==2:
        fig = plot_2dfft(dft_data, title, show, ax)
    return fig

def plot_1dfft(dft_data, title="Spiking DFT", show=True):
    # Radar parameters
    c = 3 * 10**2   # [m/us]
    f_max = 24 / 2  # [GHz]
    S = 6.55        # [GHz/us]
    # Calculate maximum range
    d_max = (f_max*c) / (2*S)
    
    freq_bins = np.arange(0, d_max, d_max/dft_data.size)[:dft_data.size]
    # Plot results
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 5))
    format_plotting()
    ax.plot(freq_bins, dft_data)
    ax.set_xlabel("Range (m)")
    ax.set_title(title)
    plt.tight_layout()
    if show:
        plt.show()
    return fig

def plot_2dfft(dft_data, title="Spiking DFT", show=True, ax=None):
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

    if not ax:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
    else:
        fig = None
    format_plotting()
    ax.imshow(20*np.log10(dft_data), extent=[-v_max, v_max-2*v_min, 0, d_max],)
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Range (m)")
    ax.set_aspect("auto")
    ax.set_title(title)
    if show:
        plt.show()
    return fig, ax


def plot_cfar(cfar_object, title= "Spiking OS-CFAR", show=True, ax=None):
    """
    Visualize the input and output data.
    """
    if cfar_object.input_array.ndim == 1:
        fig = plot_cfar_1d(cfar_object, show, title)
    elif cfar_object.input_array.ndim == 2:
        fig = plot_cfar_2d(cfar_object, show, title, ax=ax)
    return fig

def plot_cfar_1d(cfar_object, show=True, title="OS-CFAR"):
    """
    Visualize the 1D input and output data.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 5))
    format_plotting()

    # plot input data
    if not cfar_object.zero_padding:
        ax.plot(cfar_object.input_array, label='signal')
    else:
        tmp_padding = cfar_object.guarding_cells + cfar_object.neighbour_cells
        tmp_size = cfar_object.input_array.size - 2*tmp_padding   
        ax.plot(cfar_object.input_array[tmp_padding:tmp_size+tmp_padding],
                label='signal')

    # plot threshold line
    if cfar_object.show_threshold:
        low = cfar_object.guarding_cells+cfar_object.neighbour_cells
        high = cfar_object.guarding_cells+cfar_object.neighbour_cells+ \
                cfar_object.threshold.size
        # ax.plot(range(low,high),cfar_object.threshold, ls='dotted', lw=1,
        #         c='C3', label='threshold')

    # plot detected peaks
    cntr = 0
    if not cfar_object.zero_padding:
        for x in np.where(cfar_object.results>=1)[0]:
            if cntr == 0:
                plt.axvline(x+cfar_object.guarding_cells+ \
                            cfar_object.neighbour_cells,ls='--',
                            c='C1', lw=1, label='detected peaks')
                cntr +=1
            else:
                ax.axvline(x+cfar_object.guarding_cells+ \
                            cfar_object.neighbour_cells, ls='--',
                            c='C1', lw=1)
    else:
        for x in np.where(cfar_object.results>=1)[0]:
            if cntr == 0:
                plt.axvline(x, ls='--', c='C1', lw=1, label='detected peaks')
                cntr +=1
            else:
                ax.axvline(x, ls='--', c='C1', lw=1)
    
    # plot boundaries where algorithm works properly
    if not cfar_object.zero_padding:
        ax.axvline(cfar_object.guarding_cells+cfar_object.neighbour_cells-1,
                    ls='--', c='C2', lw=1, label='algorithm boundaries')
        ax.axvline(cfar_object.results.size+cfar_object.guarding_cells+
                    cfar_object.neighbour_cells,
                    ls='--', c='C2', lw=1)
    else:
        pass
    
    # show plot
    ax.legend()
    ax.set_xlabel("Range (m)")
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_title(cfar_object.name)
    if show:
        ax.grid(True)
        plt.show()

    return fig

def plot_cfar_2d(cfar_object, show=True, title="Spiking DFT", ax=None):
    """
    Visualize the 2D input and output data.
    """
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

    # Create the CFAR array
    if not cfar_object.zero_padding:
        result = np.zeros_like(cfar_object.input_array)
        temp = cfar_object.guarding_cells+cfar_object.neighbour_cells
        result[temp:temp+cfar_object.results.shape[0],
                temp:temp+cfar_object.results.shape[1]] = cfar_object.results
    else:
        result = cfar_object.results

    # Plot output 2D CFAR
    if not ax:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
        plt.ylabel("Range (m)")
    else:
        fig = None
        ax.set_yticks([])
    format_plotting()
    ax.imshow(result, extent=[-v_max, v_max-2*v_min, 0, d_max])
    ax.set_aspect("auto")
    plt.xlabel("Speed (m/s)")
    plt.title(title)
    if show:
        plt.show()

    return fig, ax
