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
    plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
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

def get_range_limits(dims, nsamples, nchirps=128, T_c=54, c=300, f_0=77,
                     f_s=24.6, S=6.5):
    """
    Return the range and velocity scales, based on radar parameters

    @param nsamples: number of samples per chirp
    @param nchirps: number of chirps in a frame
    @param T_c: chirp period, in [us]
    @param c: speed of light, in [m/us]
    @param f_0: chirp starting frequency, in [GHz]
    @param f_s: ADC sampling rate, in [MHz]
    @param S: chirp ramp slope, in [MHz/us]
    """
    # Radar maximum detectable frequency (in MHz) and wavelength
    f_max = f_s / 2
    wavelength = c*1000 / f_0
    # Calculate maximum and minimum measurable ranges
    d_max = (f_max*c) / (2*S)
    d_min = d_max / (nsamples)
    # Calculate maximum and minimum measurable velocities
    v_max = wavelength / (4*T_c)
    v_min = v_max / nchirps
    if dims==1:
        return (d_max, d_min)
    else:
        return (d_max, d_min, v_max, v_min)

def plot_dft(dft_data, title, show=True, ax=None, cropped=False):
    if dft_data.ndim==1:
        fig, ax = plot_1dfft(dft_data, title, show, cropped)
    elif dft_data.ndim==2:
        fig, ax = plot_2dfft(dft_data, title, show, ax, cropped)
    return (fig, ax)

def plot_1dfft(dft_data, title="Spiking DFT", show=True, cropped=False):
    format_plotting()
    plt.close()
    d_max, d_min = get_range_limits(dims=1, nsamples=dft_data.size)
    freq_bins = np.arange(d_min, d_max+d_min, d_min)[:dft_data.size]
    if cropped:
        dft_data = dft_data[:200]
        freq_bins = freq_bins[:200]
    # Plot results
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 5))
    ax.plot(freq_bins, dft_data)
    ax.set_xlabel("Range (m)", fontsize=20)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_title(title)
    plt.tight_layout()
    if show:
        plt.show()
    return (fig, ax)

def plot_2dfft(dft_data, title="Spiking DFT", show=True, ax=None,
               cropped=False):
    d_max, d_min, v_max, v_min = get_range_limits(dims=2, nchirps=128,
                                                  nsamples=dft_data.shape[0])
    if cropped:
        dft_data = dft_data[-200:, :]
        d_max *= 200/449
    if not ax:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
    else:
        fig = None
    ax.imshow(20*np.log10(dft_data),
              extent=[-v_max, v_max-2*v_min, d_min, d_max])
    format_plotting()
    ax.set_xlabel("Speed (m/s)", fontsize=20)
    ax.set_ylabel("Range (m)", fontsize=20)
    ax.set_aspect("auto")
    ax.set_title(title)
    plt.tight_layout()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    if show:
        plt.show()

    return fig, ax

def plot_cfar(cfar_object, title= "Spiking OS-CFAR", show=True, ax=None,
              cropped=False):
    """
    Visualize the input and output data.
    """
    if cfar_object.input_array.ndim == 1:
        fig, ax = plot_cfar_1d(cfar_object, show, title, cropped)
    elif cfar_object.input_array.ndim == 2:
        fig, ax = plot_cfar_2d(cfar_object, show, title, ax=ax)
    return (fig, ax)

def plot_cfar_1d(cfar_object, show=True, title="OS-CFAR", cropped=False):
    """
    Visualize the 1D input and output data.
    """
    format_plotting()
    d_max, d_min = get_range_limits(dims=1, nsamples=449)
    freq_bins = np.arange(d_min, d_max+d_min, d_min)
    if cropped:
        freq_bins = freq_bins[:200]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 5))

    # plot input data
    if not cfar_object.zero_padding:
        ax.plot(cfar_object.input_array, label='FT')
    else:
        tmp_padding = cfar_object.guarding_cells + cfar_object.neighbour_cells
        tmp_size = cfar_object.input_array.size - 2*tmp_padding   
        ax.plot(freq_bins,
                cfar_object.input_array[tmp_padding:tmp_size+tmp_padding],
                label='FT')

    # plot threshold line
    if cfar_object.show_threshold:
        low = cfar_object.guarding_cells+cfar_object.neighbour_cells
        high = cfar_object.guarding_cells+cfar_object.neighbour_cells+ \
                cfar_object.threshold.size

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
                plt.axvline(x+0.63, ls='--', c='C1', lw=1, label='detected peaks')
                cntr +=1
            else:
                ax.axvline(x*0.632+0.63, ls='--', c='C1', lw=1)
    
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
    ax.legend(framealpha=1)
    ax.set_xlabel("Range (m)", fontsize=20)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(title)
    plt.tight_layout()
    if show:
        ax.grid(True)
        plt.show()

    return (fig, ax)

def plot_cfar_2d(cfar_object, show=True, title="Spiking DFT", ax=None):
    """
    Visualize the 2D input and output data.
    """
    d_max, d_min, v_max, v_min = get_range_limits(dims=2, nchirps=128,
                                                  nsamples=449)

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
    ax.imshow(result, extent=[-v_max, v_max-2*v_min, d_min, d_max])
    ax.set_aspect("auto")
    plt.xlabel("Speed (m/s)", fontsize=20)
    plt.title(title)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    if show:
        plt.show()

    return fig, ax
