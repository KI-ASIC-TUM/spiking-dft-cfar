from numba import jit

# outsourced IF simulation to efficient numba compiled function
@jit(nopython=True)
def simulate_IF_neuron(start,end,t_step,spike_times,weights):
    '''
    Simulation of IF neuron. Simultates neuron behaviour from <start> to <end>
    using timesteps of size <t_step>. <spike_times> defines the spike times of 
    the pre-synaptic neurons. 

    @param start: float, starting time of simulation
    @param end: float, end time of simulation
    @param t_step: float, time steps during simulation
    @param spike_times: np.array, spike times (one per presynaptic neuron)
    @param weights: np.array, connection weights
    '''
    time = start
    res = 0
    v_mem = 0.0
    while (time < end + 2*t_step):
        
        # add spikes weights of spiking inputs to v_mem
        v_mem += weights[(time<=spike_times) & 
                          (spike_times<time+t_step)].sum()

        # update time
        time += t_step
    
        # test for spike
        if v_mem >= 1.0:
            res = 1
            break
    return res
