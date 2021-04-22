#!/usr/bin/env python3
"""
Main file for running the spiking_dft_cfar library
"""
# Standard libraries
import argparse
import json
import logging
import pathlib
import time
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
    parser.add_argument("-s", type=str2bool, default=False,
                        nargs='?', const=True, metavar="",
                        help="Show the plot after the simulation"
                       )
    # Get the values from the argument list
    args = parser.parse_args()
    config_file = args.config_file
    dimensions = args.dimensions
    method = args.method
    from_file = args.f
    show_plot = args.s
    return (config_file, dimensions, method, from_file, show_plot)


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
    fpath = pathlib.Path(__file__).resolve().parent.joinpath(fname)
    # Load the general parameters
    dft_args = {}
    cfar_args = config_data["cfar_args"]["{}D".format(dims)]
    # Append encoding parameteres if an SNN is used
    if method=="SNN":
        cfar_args.update(config_data["cfar_encoding_parameters"])
        dft_args = config_data["dft_encoding_parameters"]
    return (fpath, cfar_args, dft_args)


def conf_logger():
    # Create log folder
    logpath = pathlib.Path(__file__).resolve().parent.joinpath("log")
    pathlib.Path(logpath).mkdir(parents=True, exist_ok=True)
    datetime = time.strftime("%Y-%m-%d %H:%M:%S")
    fdatetime = time.strftime("%Y%m%d-%H%M%S")
    # Create logger
    logger = logging.getLogger('S-DFT S-CFAR')
    logger.setLevel(logging.DEBUG)

    # Create file handler
    file_handler = logging.FileHandler("{}/{}.log".format(logpath, fdatetime))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  "%H:%M:%S")
    file_handler.setFormatter(formatter)
    file_handler.stream.write("{} MAIN PROGRAM EXECUTION\n".format(datetime))
    logger.addHandler(file_handler)

    # Create console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def run(fpath, dims, dft_args, cfar_args, method, from_file, show_plot,
        cropped, fmt):
    """
    Run the algorithm with the loaded configuration
    """
    # Only the 900 first samples contain information
    data_cube = snn_dft_cfar.utils.read_data.bbm_get_datacube(fpath)[:, :900]
    # Run corresponding routine based on the number of dimensions
    if dims==1:
        chirp_n = 77
        raw_data = data_cube[chirp_n]
    if dims==2:
        raw_data = data_cube
    dft, cfar = snn_dft_cfar.run_dft_cfar.dft_cfar(raw_data, dims, dft_args,
                                                   cfar_args, method, from_file,
                                                   cropped)
    snn_dft_cfar.run_dft_cfar.plot(dft, cfar, dims, method, show=show_plot,
                                   fmt=fmt, cropped=cropped)
    return


def main():
    """
    Run the DFT and CFAR on BBM data
    """
    cropped = True
    fmt = "pdf"
    conf_file, dims, method, from_file, show_plot = parse_args()
    fpath, cfar_args, dft_args = load_config(conf_file, dims, method)
    logger = conf_logger()

    init_message = "Running spiking-dft-cfar program:"
    init_message +="\n- Configuration file: {}".format(conf_file)
    init_message +="\n- Number of dimensions: {}".format(dims)
    init_message +="\n- Method: {}".format(method)
    logger.info(init_message)

    run(fpath, dims, dft_args, cfar_args, method, from_file, show_plot,
        cropped, fmt)
    return


if __name__ == "__main__":
    main()
