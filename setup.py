#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="spiking-dft-cfar",
    version="0.1",
    description="Package for performing 2D-DFT and OS-CFAR on radar signals",
    url="https://gitlab.lrz.de/ki-asic/spiking-dft-cfar",
    author="Technical University of Munich. Informatik VI",
    packages=find_packages(exclude=["examples"]),
    install_requires=[
        "numpy>=1.16",
        "matplotlib>=3.1.2",
    ],
)
