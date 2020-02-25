# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 12:46:36 2020
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# LOAD DATA
domainData = np.loadtxt('domain_y.dat', unpack=1) #unpack=1 to transpose the read in data in order to turn columns into arrays
densData = np.loadtxt('dens0000050.txt', unpack=0)
a = 4*np.pi*domainData**2
rho = densData

y = domainData * densData

# PLOT DATA
plt.figure(figsize=(10, 10))

plt.plot(domainData, densData, 'b.') # x = log of rotation velocity, y = log of baryonic mass
#plt.plot(densData, domainData, 'g.')
#plt.xlim(1.4,2.7)
#plt.ylim(8,11.7)


# FIT A SLOPE TO DATA
#a = observationData[1]
#b = observationData[0]
#slope, intercept, r_value, p_value, std_err = stats.linregress(a, b)

#fit = np.array(intercept + slope * a)

#plt.plot(a, fit)


# DESIGN PLOT
#plt.xlabel("Log of rotation velocity [km/s]")
#plt.ylabel("Log of baryonic mass [M_sun]")
#plt.title('Baryonic Tully-Fisher Relation')
#plt.legend(loc="best", fontsize=20)
#print('slope/exponent =', slope)
#plt.text(1.5, 11.0, "slope/exponent = 3.6")

plt.savefig('Einarbeitung01.PNG')
print(np.dot(a, rho))
#plt.show()