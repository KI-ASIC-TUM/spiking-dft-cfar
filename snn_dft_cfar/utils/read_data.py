#!/usr/bin/env python3
"""
Module containing functions for reading radar data
"""
# Standard libraries
import numpy as np
# Local libraries


def soli_get_datacube(filename):
    """
    Read file with Soli sensor data and arrange it in a data cube
    """
    with open(filename, "r") as f:
        data = f.readlines()
    # Remove first line, containing header
    data.pop(0)

    # Go through all lines in the text file and parse data
    data_cube = None
    for row in data:
        # Ignore line if it only contains a newline character
        if row == "\n":
            continue
        # Separate items with a white space in between, and remove newline chars
        row_list = row.split(" ")
        try:
            row_list.remove("\n")
        except ValueError:
            pass
        # Construct the data cube. First iteration creates an array
        # following iterations stack on top
        if data_cube is None:
            data_cube = np.array(row_list)
            samples_per_chirp = len(row_list)
        elif len(row_list) == samples_per_chirp:
            data_cube = np.vstack((data_cube, row_list))
        data_cube = data_cube.astype(np.float)
    return data_cube

def bbm_get_datacube(filename):
    """
    Read file with the BBM simulator data and arrange it in a data cube
    """
    data_cube = None
    with open(filename, "r") as f:
        data = f.readlines()
    for row in data:
        row = row.split(",")
        row[-1] = row[-1].strip("\n")
        row_arr = np.array(row).astype(np.float)
        # Construct the data cube. First iteration creates an array
        # following iterations stack on top
        if data_cube is None:
            data_cube = np.array(row_arr)
        else:
            data_cube = np.vstack((data_cube, row_arr))
    data_cube = data_cube.transpose()
    return data_cube
