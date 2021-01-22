#!/usr/bin/env python3
"""
Main file for running the spiking_dft_cfar library
"""
# Standard libraries
import argparse
import json
# Local libraries
import snn_dft_cfar.run_dft_cfar


def parse_args():
    """
    Obtain the simulation options from the input arguments
    """
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser(
        usage="main.py [-h] [-d {1 2}] [-m {numpy SNN}] [-f] config_file"
    )
    parser.add_argument("config_file", type=str,
                        help="Relative location of the configuration file"
                       )
    parser.add_argument("-d", "--dimensions", type=int, choices=[1, 2],
                        default=1, metavar="",
                        help="{1 | 2} number of DFT dimensions"
                       )
    parser.add_argument("-m", "--method", type=str, choices=["numpy", "SNN", "ANN"],
                        default="SNN", metavar="",
                        help="{numpy | SNN} method used for running the system"
                       )
    parser.add_argument("-f", type=str2bool, default=False,
                        nargs='?', const=True, metavar="",
                        help="Get the S-DFT data from a local file"
                       )
    # Get the values from the argument list
    args = parser.parse_args()
    config_file = args.config_file
    dimensions = args.dimensions
    method = args.method
    from_file = args.f
    return (config_file, dimensions, method, from_file)


def load_config(conf_file, dims, method):
    """
    Load the configuration file with the simulation parameters

    @param conf_file: str with the relative address of the config file
    @param dims: Number of Fourier dimensions of the experiment 
    """
    # Load configuaration data from local file
    with open(conf_file) as f:
        config_data = json.load(f)
    fname = config_data["filename"]
    # Load the general parameters
    dft_args = {}
    cfar_args = config_data["cfar_args"]["{}D".format(dims)]
    # Append encoding parameteres if an SNN is used
    if method=="SNN":
        cfar_args.update(config_data["cfar_encoding_parameters"])
        dft_args = config_data["dft_encoding_parameters"]
    return (fname, cfar_args, dft_args)


def main():
    """
    Run the DFT and CFAR on BBM data
    """
    conf_file, dims, method, from_file = parse_args()
    fname, cfar_args, dft_args = load_config(conf_file, dims, method)
    # Only the 900 first samples contain information
    data_cube = snn_dft_cfar.utils.read_data.bbm_get_datacube(fname)[:, :900]
    # Run corresponding routine based on the number of dimensions
    if dims==1:
        chirp_n = 15
        raw_data = data_cube[chirp_n]
    if dims==2:
        raw_data = data_cube
    dft, cfar = snn_dft_cfar.run_dft_cfar.dft_cfar(raw_data, dims, dft_args,
                                                   cfar_args, method, from_file)
    snn_dft_cfar.run_dft_cfar.plot(dft, cfar, dims, method)
    return


if __name__ == "__main__":
    main()
