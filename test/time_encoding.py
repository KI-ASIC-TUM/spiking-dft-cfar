import numpy as np
from snn_dft_cfar.utils.encoding import TimeEncoder
import matplotlib.pyplot as plt

# define parameters
t_max=50
t_min=0
x_max=5
x_min=0

# print config
print(20*'=','TimeEncoding Test',20*'=')
print('t_max={}'.format(t_max))
print('t_min={}'.format(t_min))
print('x_max={}'.format(x_max))
print('x_min={}'.format(x_min))

# initialize time encoder
time_encoder = TimeEncoder(t_max=t_max,t_min=t_min,x_max=x_max,x_min=x_min)

# define an array to encode
to_encode = np.arange(0,5,0.01)

# encode array
encoded = time_encoder(to_encode)

# print information
print("""
Output graph should show a linear function with negative
nslope. The points (x_min,t_max) and (x_max,t_min) define
the linear function.""")
print(59*'=','\n')

# visualize encoding function
plt.plot(to_encode,encoded)
plt.grid(True)
plt.xlabel('VALUES (to encode)')
plt.ylabel('TIME (encoded)')
plt.show()