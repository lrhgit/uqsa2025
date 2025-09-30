'''
Created on Dec 1, 2016

@author: leifh
'''
# matplotlib and settings
import matplotlib.pyplot as plt

# import modules
import sys
sys.path.append('./python_source/') 
import numpy as np
from numpy import linalg as LA
import chaospy as cp
from monte_carlo import generate_sample_matrices_mc
from monte_carlo import calculate_sensitivity_indices_mc
import pandas as pd
from operator import index

# end import modules

# start the linear model
def test_function(w, z):
    return np.sum(w*z, axis=1)
# end the linear model

# start the C1 testfunction 
def test_function_C1(Z):
    Ns,n = Z.shape
    Zprod=Z[:,0]
    
    for i in range(1,n):
        Zprod=np.multiply(Zprod,Z[:,i])
        
    Zprod*=2**n
    return Zprod 
# End the C1 testfunction 

# start the B2 testfunction 

def test_function_B2(Z):
    ai=50
    Ns,n = Z.shape
    Z=np.abs(4*Z-2)+ai
    Z=Z/(1+ai)

    Zprod=Z[:,0]

    for i in range(1,n):
        Zprod=np.multiply(Zprod,Z[:,i])


    return Zprod
# End the B2 testfunction

# start the B2 analytical testfunction 
def analytical_Si_B2(Nrv):
    ai=50.0
    
    Vi = 1.0/(1.0+ai)**2/3.0*np.ones(Nrv)
    
    V=1+Vi[0]

    for i in range(1,Nrv):
        V=np.multiply(V,1+Vi[i])
    
    V=V-1
    Si=Vi/V
    
    return Si

# end the B2 analytical testfunction
 
def main():
    # Define the random variables

    Nrv = 7  # number of random variables 
#     zm = np.array([[10., i] for i in range(1, Nrv + 1)])
    # Generate distributions for each element in z and sample
    Ns = 1000000
        
    pdfs = []

#     for i, z in enumerate(zm):
#         pdfs.append(cp.Normal(z[0], z[1]))
    
    for i in range(Nrv):
        pdfs.append(cp.Uniform(0,1))
        
    jpdf = cp.J(*pdfs)
    
    # generate Z
    Z = jpdf.sample(Ns)
    
    # End definition of the random variables
       
    # Scatter plot section
    make_plot=0 # set unequal to zero for scatter plot
                # works for Nrv=4 only
    if (make_plot):
        # Evaluate the model for all samples
        Y = test_function_C1(Z.transpose())
        
        # Scatter plots of data for visual inspection of sensitivity
        fig=plt.figure()
        for k in range(Nrv):
            plt.subplot(2, 2, k + 1)
            plt.plot(Z[k, :], Y[:], '.')
            xlbl = 'Z' + str(k)
            plt.xlabel(xlbl)
            
        fig.tight_layout()  # adjust subplot(s) to the figure area.

    # End scatter plot section


    # Monte Carlo section 
    # get joint distributions
   
    Ns_mc = 500000
    #Nrv = len(jpdf) # Number of random variables
    sample_method='R'
    
    # 1. Generate sample matrices
    A, B, C = generate_sample_matrices_mc(Ns_mc, Nrv, jpdf, sample_method)

    
    #test_function_list=[test_function_C1,test_function_B2]
    test_function_list=[test_function_B2]
            
    Y_C = np.empty((Ns_mc,Nrv))
    
    for test_function in test_function_list:
        Y_A,Y_B = [test_function(A), test_function(B)]
        
        for i in range(Nrv):
            Ci = C[i, :, :]
            
            Y_C[:,i]=test_function(Ci)
        print('Monte Carlo sampling')
        S_mc, ST_mc = calculate_sensitivity_indices_mc(Y_A, Y_B, Y_C)  
        
        # print mc-estimates and analytical Sobol indices
        S_a=analytical_Si_B2(Nrv)
        Sensitivities=np.column_stack((S_mc,S_a))
        row_labels= ['S_'+str(idx) for idx in range(1,Nrv+1)]
        print("First Order Indices")
        print(pd.DataFrame(Sensitivities,columns=['Smc','Sa'],index=row_labels).round(2))
         
                
    # End Monte Carlo 

    if(make_plot):
        plt.show()
        plt.close()
        
    
        

    # Polychaos computations
    polynomial_order = 4
    poly = cp.orth_ttr(polynomial_order, jpdf)
    Ns_pc = len(poly)*2 #len(poly) gives the number of coefficients
    samples_pc = jpdf.sample(Ns_pc)
    Y_pc = test_function_B2(samples_pc.transpose())
    approx = cp.fit_regression(poly, samples_pc, Y_pc, rule="LS") #rule="T" does Tikhonov regularisation which takes a lot more time.

    exp_pc = cp.E(approx, jpdf)
    std_pc = cp.Std(approx, jpdf)
    print("Statistics polynomial chaos\n")
    print('\n        E(Y)  |  std(Y) \n')
    print('pc  : {:2.5f} | {:2.5f}'.format(float(exp_pc), std_pc))
    
    print('before approx')
    
    S_pc = cp.Sens_m(approx, jpdf)
    print('after approx')
    Sensitivities=np.column_stack((S_mc,S_pc, S_a))
    print("\nFirst Order Indices")
    print(pd.DataFrame(Sensitivities,columns=['Smc','Spc','Sa'],index=row_labels).round(3))

#     print("\nRelative errors")
#     rel_errors=np.column_stack(((S_mc - s**2)/s**2,(S_pc - s**2)/s**2))
#     print(pd.DataFrame(rel_errors,columns=['Error Smc','Error Spc'],index=row_labels).round(3))

    print('stopped')
    sys.exit()

    # Polychaos convergence
    Npc_list = np.logspace(1, 3, 10).astype(int)
    error = []

    for i, Npc in enumerate(Npc_list):
        Zpc = jpdf.sample(Npc)
        Ypc = test_function(w, Zpc.T)
        Npol = 4
        poly = cp.orth_chol(Npol, jpdf)
        approx = cp.fit_regression(poly, Zpc, Ypc, rule="T")
        s_pc = cp.Sens_m(approx, jpdf)
        error.append(LA.norm((s_pc - s**2)/s**2))

    plt.figure()
    plt.semilogy(Npc_list, error)
    _=plt.xlabel('Nr Z')
    _=plt.ylabel('L2-norm of error in Sobol indices')

    # # Scatter plots of data, z-slices, and linear model
    fig=plt.figure()

    Ndz = 10  # Number of slices of the Z-axes

    Zslice = np.zeros((Nrv, Ndz))  # array for mean-values in the slices
    ZBndry = np.zeros((Nrv, Ndz + 1))  # array for boundaries of the slices
    dz = np.zeros(Nrv)

    for k in range(Nrv):
        plt.subplot(2, 2, k + 1)

        zmin = np.min(Z[k, :])
        zmax = np.max(Z[k, :])  # each Z[k,:] may have different extremas
        dz[k] = (zmax - zmin) / Ndz

        ZBndry[k, :] = np.linspace(zmin, zmax, Ndz + 1) # slice Zk into Ndz slices
        Zslice[k, :] = np.linspace(zmin + dz[k] / 2., zmax - dz[k] / 2., Ndz) # Midpoint in the slice

        # Plot the the vertical slices with axvline
        for i in range(Ndz):
            plt.axvline(ZBndry[k, i], np.amin(Y), np.amax(Y), linestyle='--', color='.75')

        # Plot the data
        plt.plot(Z[k, :], Y[:], '.')
        xlbl = 'Z' + str(k)
        plt.xlabel(xlbl)
        plt.ylabel('Y')

        Ymodel = w[k] * Zslice[k, :]  # Produce the straight line

        plt.plot(Zslice[k, :], Ymodel)

        ymin = np.amin(Y); ymax = np.amax(Y)
        plt.ylim([ymin, ymax])
    
    fig.tight_layout()  # adjust subplot(s) to the figure area.    

    # # Scatter plots of averaged y-values per slice, with averaged data

    Zsorted = np.zeros_like(Z)
    Ysorted = np.zeros_like(Z)
    YsliceMean = np.zeros((Nrv, Ndz))

    fig=plt.figure()
    for k in range(Nrv):
        plt.subplot(2, 2, k + 1)

        # sort values for Zk, 
        sidx = np.argsort(Z[k, :]) #sidx holds the indexes for the sorted values of Zk
        Zsorted[k, :] = Z[k, sidx].copy()
        Ysorted[k, :] = Y[sidx].copy()  # Ysorted is Y for the sorted Zk

        for i in range(Ndz):
            plt.axvline(ZBndry[k, i], np.amin(Y), np.amax(Y), linestyle='--', color='.75')

            # find indexes of z-values in the current slice
            zidx_range = np.logical_and(Zsorted[k, :] >= ZBndry[k, i], Zsorted[k, :] < ZBndry[k, i + 1])

            if np.any(zidx_range):  # check if range has elements
                YsliceMean[k, i] = np.mean(Ysorted[k, zidx_range])
            else:  # set value to None if noe elements in z-slice
                YsliceMean[k, i] = None

        plt.plot(Zslice[k, :], YsliceMean[k, :], '.')
        
        

        # # Plot linear model
        Nmodel = 3
        zmin = np.min(Zslice[k, :])
        zmax = np.max(Zslice[k, :])

        zvals = np.linspace(zmin, zmax, Nmodel)
        #test_function
        Ymodel = w[k] * zvals
        plt.plot(zvals, Ymodel)

        xlbl = 'Z' + str(k)
        plt.xlabel(xlbl)

        plt.ylim(ymin, ymax)
    
    fig.tight_layout()  # adjust subplot(s) to the figure area.
    
    SpoorMan=[np.nanvar(YsliceMean[k,:],axis=0)/np.var(Y) for k in range(4)]   
    print(SpoorMan)
    # end scatter plots y-values slice
    plt.show()
    plt.close()

    
    ## alternative
    #pdfs3 = [cp.Normal(mu, sig) for (mu, sig) in zm]

if __name__ == '__main__':
    main()
