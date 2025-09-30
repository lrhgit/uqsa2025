# Interactive Gstar function python implementation




# Gstar-statistics

# import modules
import numpy as np

def Vi(ai,alphai):
    return alphai**2/((1+2*alphai)*(1+ai)**2)

def V(a_prms,alpha):
    D=1
    for ai,alphai in zip(a_prms,alpha):
        D*=(1+Vi(ai,alphai))     
    return D-1


def S_i(a,alpha):
    S_i=np.zeros_like(a)
    for i, (ai,alphai) in enumerate(zip(a,alpha)):
        S_i[i]=Vi(ai,alphai)/V(a,alpha)
    return S_i

def S_T(a,alpha):
    # to be completed
    S_T=np.zeros_like(a)
    Vtot=V(a,alpha)
    for i, (ai,alphai) in enumerate(zip(a,alpha)):
        S_T[i]=(Vtot+1)/(Vi(ai,alphai)+1)*Vi(ai,alphai)/Vtot
    return S_T

# End Gstar-statistics

# Interactive plotting
from slider_helpers import build_slider_interface
from slider_helpers import make_slider_dict
from ipywidgets import FloatSlider
from plotting import plot_sobol_indices


f, ax = plt.subplots(1,1)
f.suptitle('G* function with variable coefficients')


def model(**kwargs):
    Nk = len(kwargs) // 3

    a = [kwargs[f'a{i}'] for i in range(1, Nk + 1)]
    alpha = [kwargs[f'alpha{i}'] for i in range(1, Nk + 1)]
    delta = [kwargs[f'delta{i}'] for i in range(1, Nk + 1)]

    Si = S_i(a, alpha)
    ST = S_T(a, alpha)

    plot_sobol_indices(Si, ST, ax)

    
prm_strn=['a', 'alpha', 'delta'];
nk=4
slider_dict = make_slider_dict(prm_strn, nk)

build_slider_interface(slider_dict, model, sliders_per_row=nk)

# End interactive plotting
