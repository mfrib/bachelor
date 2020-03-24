import astropy.constants as c
import numpy as np
from simple_slider import get_disk_height, get_surface_density

au = c.au.cgs.value


def logp(params, x_data, y_data, n_planets):

    x_data, alpha, sig0, p, R_p, h_p, mass_ratios = params_format(params, x_data, y_data, n_planets)

    # construct the model
    sig_model = get_surface_density(x_data, alpha, sig0, p, R_p, h_p, mass_ratios)
    sigma2    = (0.05 * y_data)**2

    # calculate logP

    # logP = -0.5 * np.sum((y_data[0:-1] - sig_model[0:-1]) ** 2 / sigma2[0:-1] + np.log(2 * np.pi * sigma2[0:-1]))
    logP = -0.5 * np.sum((y_data[0:-1] - sig_model[0:-1]) ** 2 / sigma2[0:-1])

    return logP


def params_format(params, x_data, y_data, n_planets):

    # convert parameters from list to correct input format for get_surface_density
    
    alpha = params[0]
    sig0  = params[1]
    p     = params[2]
    R_p   = []
    mass_ratios = []
    h_p = []
    
    for n in range(n_planets):
        R_p         += [params[3 + 2 * n]]
        mass_ratios += [params[4 + 2 * n]]
        #h_p += [np.interp(R_p, x_data, get_disk_height(x_data))[n]]
    """
    R_p += [params[3]]
    mass_ratios += [params[4]]
    if n_planets == 3:
        R_p += [params[5]]
        R_p += [params[7]]
        mass_ratios += [params[6]]
        mass_ratios += [params[8]]
    """    
        
    h_p = np.interp(R_p, x_data, get_disk_height(x_data))
    
    return x_data, alpha, sig0, p, R_p, h_p, mass_ratios


def conv_values(params, x_data, n_planets):
    
    x_min = np.log10(x_data[0]/ au) 
    x_max = np.log10(x_data[-1]/ au) 
    
    params = params.T
    
    # convert walkers from {0,1} to physical values for now    
    params[0] = 10**(params[0]*3 -4)
    params[1] = 10**(4 * (params[1]-0.5))
    params[2] = (2.0 * params[2]) - 2
    for n in range(n_planets):
        params[3+2*n] = 10**(params[3 + 2 * n] * (x_max - x_min) + x_min) * au
        params[4+2*n] = 10**(params[4 + 2 * n] * 2.0 - 4.0)
    
    return params.T

def log_prior(params, x_data, n_planets, masks):
    mask_max = masks[0]
    mask_min = masks[1]
    #mask_max = conv_values(np.ones_like(params), x_data, n_planets)
    #mask_min = conv_values(np.zeros_like(params), x_data, n_planets)
    
    if n_planets == 3: 
        R_p1, R_p2, R_p3 = params[3]*1.3, params[5], params[7]*0.7
    else:
        R_p1, R_p2, R_p3 = 1, 2, 3

    if np.all(np.array(params) < mask_max) and np.all(np.array(params) > mask_min):
        return 0.0
    return -np.inf

def log_prob(params, x_data, y_data, n_planets, masks):
    lp = log_prior(params, x_data, n_planets, masks)
    if not np.isfinite(lp):
        return -np.inf
    return logp(params, x_data, y_data, n_planets)
