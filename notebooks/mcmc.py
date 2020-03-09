#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np  
import matplotlib.pyplot as plt  
import astropy.constants as c  
import astropy.units as u  

year = (1*u.year).cgs.value
au   = c.au.cgs.value

from simple_slider import Widget
from simple_slider import kanagawa_profile
from simple_slider import get_surface_density

get_ipython().run_line_magic('matplotlib', 'notebook')


# In[ ]:


w = Widget()


# Get the data from the widget

# In[ ]:


time   = w.t

t = time.searchsorted(1e6 * year)

r   = w.r
sig = w.sigma[:, t]


# Plot the data

# In[ ]:


f, ax = plt.subplots()
ax.loglog(r / au, sig)


# Define the logp function: create a model based on the parameters and compare it to the data

# In[ ]:


def logp(params, x_data, y_data, n_planets):
    
    # convert parameter to physical values
    
    alpha = params[0]
    ...
    
    # construct the model
    
    sig_model = get_surface_density(x_data, alpha, sig0, p, R_p, h_p, mass_ratios)
    
    # calculate logP
    
    logP = ...
    
    return logP


# Test if logP works

# In[ ]:


logp([10, -0.9, 1e3, 60* au, M_jup], r, sig, 1)
# should return a number


# Now optimize it 

# In[ ]:


get_ipython().system('jupyter nbconvert --to python mcmc.ipynb')

