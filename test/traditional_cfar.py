import numpy as np
from snn_dft_cfar.cfar import CACFAR

# define parameters
data_size = 50
data_peaks = 3 
guarding_cells = 2
neighbour_cells = 7
scale_factor = 0.65

# generate data with peaks
rng = np.random.default_rng()
data = rng.random(data_size)
for i in rng.integers(data_size, size=data_peaks):
    data[i] += 1

# initialize CFAR 
cacfar = CACFAR(scale_factor,guarding_cells,neighbour_cells)

# run CFAR
cacfar(data)

# visualize CFAR
cacfar.plot()