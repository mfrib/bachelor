import astropy.constants as c
import numpy as np
from simple_slider import get_disk_height, get_surface_density

au = c.au.cgs.value


def logp(params, x_data, y_data, n_planets):

    x_data, alpha, sig0, p, R_p, h_p, mass_ratios = params_format(params, x_data, y_data, n_planets)

    # construct the model
    model     = get_surface_density(x_data, alpha, sig0, p, R_p, h_p, mass_ratios)
    sig_model = model['sigma']
    R1        = model['R1']
    sigma2    = (0.05 * y_data)**2

    # exclude region around planet(s)

    #mask = (np.abs(x_data[:, None] - R_p[None, :]) > (R1 * R_p)[None, :]).all(1)
    mask = R1

    # calculate logP

    #logP = -0.5 * np.sum((y_data[mask] - sig_model[mask]) ** 2 / sigma2[mask])
    logP = -0.5 * np.sum((y_data - sig_model) ** 2 / sigma2)

    return logP


def params_format(params, x_data, y_data, n_planets):

    # convert parameters from nparray to correct input format for get_surface_density

    alpha = params[0]

    sig0  = params[1]
    p     = params[2]
    R_p   = []
    mass_ratios = []
    #h_p = []
    """
    R_p = params[3::2]
    mass_ratios = params[4::2]
    """
    for n in range(n_planets):
        R_p         += [params[3 + 2 * n]]
        mass_ratios += [params[4 + 2 * n]]
        #h_p += [np.interp(R_p, x_data, get_disk_height(x_data))[n]]
    """

    other_params = params[3:].reshape(n_planets, 2)
    R_p = other_params[:,0]
    mass_ratios = other_params[:,1]
    """

    #h_p = np.interp(R_p, x_data, get_disk_height(x_data))
    h_p = get_disk_height(np.array(R_p))
    
    return x_data, alpha, sig0, p, R_p, h_p, mass_ratios


def conv_values(params, x_data, n_planets):

    x_min = np.log10(x_data[0]/ au)
    x_max = np.log10(x_data[-1]/ au)

    params = params.T

    # convert walkers from {0,1} to physical values

    params[0] = 10**(params[0] * 4.0 - 5.0)

    params[1] = 10**(4 * (params[1]-0.5))
    params[2] = (1.5 * params[2])

    for n in range(n_planets):
        params[3+2*n] = 10**(params[3 + 2 * n] * (x_max - x_min) + x_min) * au
        params[4+2*n] = 10**(params[4 + 2 * n] * 2.5 - 4.5)

    return params.T


def log_prior(params, x_data, n_planets, masks):
    #masks = [np.array([1.00000000e-01, 1.00000000e+02, 0.00000000e+00, 5.98391483e+15, 1.00000000e-02]),np.array([ 1.00000000e-04,  1.00000000e-02, -2.00000000e+00,  2.99195741e+14, 1.00000000e-04])]
    mask_max = masks[0]
    mask_min = masks[1]
    
    if n_planets == 3:
        R_p1, R_p2, R_p3 = params[3], params[5], params[7]
        #np.append(mask_max,[mask_max[3,4],[mask_max[3,4]])
        #np.append(mask_min,[mask_min[3,4],[mask_min[3,4]])
    else:
        R_p1, R_p2, R_p3 = 7, 10, 15
    """ """

    if np.all(np.array(params) < mask_max) and np.all(np.array(params) > mask_min) and 0.5*R_p2 < R_p1 < 0.8*R_p2 and 1.25*R_p2 < R_p3 < 2*R_p2:
        return 0.0
    return -np.inf

def log_prob_(params, x_data, y_data, n_planets, masks):
    lp = log_prior(params, x_data, n_planets, masks)
    if not np.isfinite(lp):
        return -np.inf
    return logp(params, x_data, y_data, n_planets)


def log_prob_alpha(params, x_data, y_data, n_planets, masks, alpha_input):
    params = np.insert(params, 0, alpha_input)
    lp = log_prior(params, x_data, n_planets, masks)
    if not np.isfinite(lp):
        return -np.inf
    return logp(params, x_data, y_data, n_planets)
