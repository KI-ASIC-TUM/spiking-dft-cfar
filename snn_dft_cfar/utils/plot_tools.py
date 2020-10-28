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
