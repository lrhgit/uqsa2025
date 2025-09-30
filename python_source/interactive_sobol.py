import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import sobol_seq

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
f.suptitle('Sobol sequences versus random sampling')

def update_Sobol(N):
    ax1.clear()
    ax2.clear()
    Sobol_sample = sobol_seq.i4_sobol_generate(2, 2**N)
    numpy_sample = np.random.rand(2**N, 2)

    ax1.scatter(Sobol_sample[:,0],Sobol_sample[:,1],color='blue')
    ax2.scatter(numpy_sample[:,0],numpy_sample[:,1],color='red')
    ax1.set_title('Sobol sequences')
    ax2.set_title('Random sampling')
    
    return f
