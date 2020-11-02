import numpy as np
from snn_dft_cfar.cfar import CACFAR, OSCFAR

# define parameters
data_size = 50
data_peaks = 3 
guarding_cells = 2
neighbour_cells = 7
scale_factor = 0.5
k = 4
random_seed = 4

# print test information
print(34*"=","CFAR test", 34*"=")
print("Test to see if CFAR works correctly.")
print()
print("Parameters:")
print("number of peaks : {}".format(data_peaks))
print("guarding cells  : {}".format(guarding_cells))
print("neighbour cells : {}".format(neighbour_cells))
print("scaling factor  : {}".format(scale_factor))
print("k               : {}".format(k))
print()
print("First  plot: CACFAR")
print("Second plot: OSCFAR")
print((68+11)*"=")

# generate data with peaks
rng = np.random.default_rng(seed=random_seed)
data = rng.random(data_size)
for i in rng.integers(data_size, size=data_peaks):
    data[i] += 1

# initialize CFAR 
cacfar = CACFAR(scale_factor,guarding_cells,neighbour_cells)
oscfar = OSCFAR(scale_factor,guarding_cells,neighbour_cells,k)

# run CFAR
cacfar(data)
oscfar(data)

# visualize CFAR
cacfar.plot()
oscfar.plot()