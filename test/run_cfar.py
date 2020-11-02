import numpy as np
from timeit import default_timer as timer
from snn_dft_cfar.cfar import CACFAR, OSCFAR, OSCFAR_SNN

# define parameters
data_size = 50
data_peaks = 3 
guarding_cells = 2
neighbour_cells = 7
scale_factor = 0.5
k = 4
t_max, t_min, x_max, x_min = 50, 0, 2, 0
random_seed = 4

# print test information
print()
print(34*"=","CFAR test", 34*"=")
print("Benchmark different variations of CFAR.")
print()
print("Parameters:")
print("number of peaks : {}".format(data_peaks))
print("guarding cells  : {}".format(guarding_cells))
print("neighbour cells : {}".format(neighbour_cells))
print("scaling factor  : {}".format(scale_factor))
print("k               : {}".format(k))
print()
print("Encoding parameters:")
print("t_max : {}".format(t_max))
print("t_min : {}".format(t_min))
print("x_max : {}".format(x_max))
print("x_min : {}".format(x_min))
print()


# generate data with peaks
rng = np.random.default_rng(seed=random_seed)
data = rng.random(data_size)
for i in rng.integers(data_size, size=data_peaks):
    data[i] += 1

# initialize CFAR 
cacfar = CACFAR(scale_factor,guarding_cells,neighbour_cells)
oscfar = OSCFAR(scale_factor,guarding_cells,neighbour_cells,k)
snnoscfar = OSCFAR_SNN(scale_factor,guarding_cells,neighbour_cells,k,
                     t_max, t_min, x_max, x_min)

# run CFAR benchmarks
print('running benchmarks .. \n')
start_cacfar = timer()
cacfar(data)
end_cacfar = timer()
start_oscfar = timer()
oscfar(data)
end_oscfar = timer()
start_snnoscfar = timer()
snnoscfar(data)
end_snnoscfar = timer()
print('Benchmark results: time per sliding window')
print('CACFAR : {0:.6f}'.format((end_cacfar-start_cacfar)/cacfar.results.size))
print('OSCFAR : {0:.6f}'.format((end_oscfar-start_oscfar)/oscfar.results.size))
print('SNNCFAR: {0:.6f}'.format((end_snnoscfar-start_snnoscfar)/
                            snnoscfar.results.size))
print()

# visualize CFAR
print("First  plot:    CACFAR")
print("Second plot:    OSCFAR")
print("Third  plot: SNNOSCFAR")
print("Note that the OSCFAR and SNNOSCFAR should detect the same peaks.")
print((68+11)*"=")
print()
cacfar.plot()
oscfar.plot()
snnoscfar.plot()