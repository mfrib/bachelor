# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 12:46:36 2020
"""

import matplotlib.pyplot as plt
import numpy as np

# LOAD DATA
domainData = np.loadtxt('domain_y.dat', unpack=1) #unpack=1 to transpose the read in data in order to turn columns into arrays
densData = np.loadtxt('dens0000050.txt', unpack=0)
#array = np.linspace(1, 1024, 1024)
t = np.loadtxt('time.dat')

# AVERAGES
    # DENSITY
densData_2 = np.loadtxt('dens0000050.txt', unpack=0, skiprows=1)
densData_1 = np.loadtxt('dens0000050.txt', unpack=1, max_rows=1023)
densData_0 = (densData_1 + densData_2)/2
    # RADIAL POSITION
r_1 = np.loadtxt('domain_y.dat', unpack=1, skiprows=1)
r_2 = np.loadtxt('domain_y.dat', unpack=1, max_rows=1023)
ring = np.pi*(r_1**2 - r_2**2)

# CALCULATIONS
a = 4*np.pi*domainData**2
rho = densData


# PLOT DATA
plt.figure(figsize=(10, 10))
plt.plot(domainData, densData) 
#plt.plot(array, domainData)
#plt.plot(densData, domainData, 'g.')


# DESIGN PLOT
plt.xlabel("radiale Position der Gitterzelle [cm]")
plt.ylabel("Gasoberflächendichte [g/cm^2]")
plt.title('dens0000050')

#plt.savefig('Einarbeitung01.PNG')
plt.show()
#print("Summe der Masse der Kugelschalen: ", np.dot(a, rho), "g")
print('Masse:', np.dot(ring, densData_0), 'g')
print(t[200])