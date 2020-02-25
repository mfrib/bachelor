#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c

au = c.au.cgs.value
M_sun = c.M_sun.cgs.value

r = np.loadtxt('radius.dat')
sig_g = np.loadtxt('sigma_averaged.dat')

# calculate disk mass

#M_gas = np.trapz(2 * np.pi * r * sig_g, x=r)
#print(f'disk mass = {M_gas / M_sun :.3} M_sun')

# plot density profile

f, ax = plt.subplots()
ax.loglog(r / au, sig_g[200])
ax.set_xlabel(r'r [au]')
ax.set_ylabel(r'$\Sigma_\mathrm{g}$')
plt.show()
#f.savefig('test1')