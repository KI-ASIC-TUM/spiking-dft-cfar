#!/usr/bin/env python3
"""
Functions for running the 1D and 2D DFT and OS-CFAR

It is possible to run it both spiking and non-spiking
"""
# Standard libraries
import json
# Local libraries
import snn_dft_cfar.run_dft_cfar


def load_config(config_file, dims, method):
    """
    Load the configuration file with the simulation parameters

    @param config_file: str with the relative address of the config file
    @param dims: Number of Fourier dimensions of the experiment 
    """
    # Load configuaration data from local file
    with open(config_file) as f:
        config_data = json.load(f)
    filename = config_data["filename"]
    # Load the CFAR parameters
    cfar_args = config_data["cfar_args"]["{}D".format(dims)]
    # Append encoding parameteres if an SNN is used
    encoding_parameters = config_data["encoding_parameters"]
    if method=="SNN":
        cfar_args.update(encoding_parameters)
    return (filename, cfar_args)


def main(dims=1, method="numpy", config_file="config/scenario1_default.json"):
    filename, cfar_args = load_config(config_file, dims, method)
    # Only the 900 first samples contain information
    data_cube = snn_dft_cfar.utils.read_data.bbm_get_datacube(filename)[:, :900]
    # Run corresponding routine based on the number of dimensions
    if dims==1:
        chirp_n = 15
        raw_data = data_cube[chirp_n]
    if dims==2:
        raw_data = data_cube
    dft, cfar = snn_dft_cfar.run_dft_cfar.dft_cfar(raw_data, dims, cfar_args, method)
    snn_dft_cfar.run_dft_cfar.plot(dft, cfar, dims, method)
    return


if __name__ == "__main__":
    main()
