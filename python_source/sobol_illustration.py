'''
Updated Apr, 2018
@author: leifh
'''

%matplotlib nbagg
import ipywidgets as widgets
import matplotlib.pyplot as plt
import interactive_sobol
from interactive_sobol import update_Sobol

w_slider = widgets.IntSlider(min=1, max=12, value=2, description='Samples')
widgets.interactive(update_Sobol, N=w_slider)



