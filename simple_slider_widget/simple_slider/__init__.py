
import matplotlib.pyplot as plt
import matplotlib.widgets as w
import numpy as np
import pkg_resources
import argparse
import os
import astropy.constants as c

au    = c.au.cgs.value
G     = c.G.cgs.value
k_b   = c.k_B.cgs.value
m_p   = c.m_p.cgs.value
M_sun = c.M_sun.cgs.value
mu    = 2.3


class Widget():

    def __init__(self, data_dir=None, num_planets=None, choose=None):
        """
        Initialize the widget to compare model and data

        fname : str
            file name to the data file
        """

        # Get the data
        
        switcher = {0:"data_1_planet", 1:"data_one_planet_a1e-2_M1e-3",2:"data_one_planet_a1e-2_M4e-4",
            3:"data_one_planet_a1e-3_M1e-3",4:"data_one_planet_a1e-3_M4e-4", 
            5:"data_planets_scalefree_a1e-2_mu3e-3_r100", 6:"data_planets_scalefree_a1e-2_mu1e-3_r100", 
            7:"data_planets_scalefree_a1e-2_mu3e-4_r100", 8:"data_planets_scalefree_a1e-3_mu3e-3_r100", 
            9:"data_planets_scalefree_a1e-3_mu1e-3_r100", 10:"data_planets_scalefree_a1e-3_mu3e-4_r100",
            11:"data_planets_scalefree_a1e-4_mu3e-3_r100",12:"data_planets_scalefree_a1e-4_mu1e-3_r100", 
            13:"data_planets_scalefree_a1e-4_mu3e-4_r100"}
        
        if choose is not None:
            data_dir = switcher[choose]
        
        if data_dir is None:
            data_dir = 'data_1_planet'

        # variable to choose number of planets to model
        if num_planets is None:
            num_planets = 1

        self.planet = np.concatenate((np.full(num_planets, 1), np.full(3, 0)))


        data_dir = pkg_resources.resource_filename(__name__, data_dir)

        self.r = (np.loadtxt(os.path.join(data_dir, 'radius.dat')))
        self.sigma = np.loadtxt(os.path.join(data_dir, 'sigma_averaged.dat'), unpack=1)
        try:
            self.t = np.loadtxt(os.path.join(data_dir, 'time.dat'))
        except Exception:
            self.sigma.shape = (len(self.sigma),1)
            self.t = [0]

        # define disk properties, for now hard-coded

        self.M_star = 2.3*M_sun
        self.alpha  = 1e-3
        self.q      = 0.85
        self.T      = 20 * (self.r / (100 * au))**-self.q
        self.cs     = np.sqrt(k_b * self.T / (mu * m_p))
        self.om     = np.sqrt(G * self.M_star / self.r**3)
        self.h      = self.cs / self.om

        # Make the figure

        self.fig = plt.figure(figsize=(12, 12))
        self.ax = self.fig.add_axes([0.1, 0.35, 0.6, 0.6])
        self.ax.set_xlabel(r'r [AU]')
        self.ax.set_ylabel(r'$\Sigma_\mathrm{g}[g/cm^2]$')

        # CREATE SLIDERS

        # time slider size

        slider_x0 = self.ax.get_position().x0
        slider_y0 = 0.05
        slider_w  = self.ax.get_position().width
        slider_h  = 0.04

        # get_model sliders size

        slider_x2 = self.ax.get_position().x0
        slider_y2 = 1 - 0.1 - 19 * slider_h
        slider_w2 = self.ax.get_position().width * 0.4
        slider_h2 = 0.04

        slider_x1 = self.ax.get_position().x0 + self.ax.get_position().width * 0.55
        slider_y1 = 1 - 0.1 - 19 * slider_h
        slider_w1 = self.ax.get_position().width * 0.4
        slider_h1 = 0.04

        alpha_slider_x = self.ax.get_position().x0
        alpha_slider_y = 1 - 0.1 - 17 * slider_h
        alpha_slider_w = self.ax.get_position().width * 0.4
        alpha_slider_h = 0.04

        # Gaussian sliders size
        # 1st planet, r_0 offset
        rp1_slider_x = 1 - 2.5 * self.ax.get_position().x0
        rp1_slider_y = 1 - 0.1
        rp1_slider_w = self.ax.get_position().width * 0.35
        rp1_slider_h = 0.04

        # 1st planet, mass
        mp1_slider_x = rp1_slider_x
        mp1_slider_y = 1 - 0.1 - 2 * rp1_slider_h
        mp1_slider_w = rp1_slider_w
        mp1_slider_h = 0.04

        # 2nd planet, r_0 offset
        rp2_slider_x = rp1_slider_x
        rp2_slider_y = 1 - 0.1 - 7.5 * rp1_slider_h
        rp2_slider_w = rp1_slider_w
        rp2_slider_h = 0.04

        # 2nd planet, mass
        mp2_slider_x = rp1_slider_x
        mp2_slider_y = 1 - 0.1 - 9.5 * rp1_slider_h
        mp2_slider_w = rp1_slider_w
        mp2_slider_h = 0.04

        # 3rd planet, r_0 offset
        rp3_slider_x = rp1_slider_x
        rp3_slider_y = 1 - 0.1 - 15 * rp1_slider_h
        rp3_slider_w = rp1_slider_w
        rp3_slider_h = 0.04

        # 3rd planet, mass
        mp3_slider_x = rp1_slider_x
        mp3_slider_y = 1 - 0.1 - 17 * rp1_slider_h
        mp3_slider_w = rp1_slider_w
        mp3_slider_h = 0.04

        # useful variable
        r_max = np.log10(self.r[-1] / au)
        r_min = np.log10(self.r[0] / au)

        # slider for parameter T

        self._ax_T = self.fig.add_axes([slider_x0, slider_y0, slider_w, slider_h], facecolor="lightgoldenrodyellow")
        self._slider_T = w.Slider(self._ax_T, "T", 0, len(self.t) - 0.99, valinit=self.t[-1], valfmt='%.0f') # -0.99 instead of -1 to avoid error when there is no time set
        self._ax_T.set_title("T", fontsize='small')
        self._slider_T.on_changed(self.update)

        self.model_line_T, = self.ax.loglog(self.r / au, self.sigma[:, int(self._slider_T.val)], 'k-')

        # sliders for get_model

        # power law sliders

        self._ax_sigma = self.fig.add_axes([slider_x2, slider_y2, slider_w2, slider_h2], facecolor="darksalmon")
        self._slider_sigma = w.Slider(self._ax_sigma, "$\\Sigma_0$", -2, 2, valinit=0.43, valfmt='%.2f')
        self._ax_sigma.set_title("$\\Sigma_0$", fontsize='small')
        self._slider_sigma.on_changed(self.update)

        self._ax_Pow = self.fig.add_axes([slider_x1, slider_y1, slider_w1, slider_h1], facecolor="lightsalmon")
        self._slider_Pow = w.Slider(self._ax_Pow, "p", 0, 1.5, valinit=0.8, valfmt='%.2f')
        self._ax_Pow.set_title("Power", fontsize='small')
        self._slider_Pow.on_changed(self.update)

        self._ax_alpha = self.fig.add_axes([alpha_slider_x, alpha_slider_y, alpha_slider_w, alpha_slider_h], facecolor="mistyrose")
        self._slider_alpha = w.Slider(self._ax_alpha, "$\\alpha$", -5, -1, valinit=np.log10(self.alpha), valfmt='%.2f')
        self._ax_alpha.set_title("$\\alpha$", fontsize='small')
        self._slider_alpha.on_changed(self.update)

        self.model_line_A, = self.ax.loglog(self.r / au, np.ones_like(self.r))
        # self.planet_line,   = self.ax.loglog(self.r / au, np.ones_like(self.r)*2)

        # Planet sliders 1
        if self.planet[0] == 1:

            # planet Position in units of log10(AU)
            self._ax_rp1 = self.fig.add_axes([rp1_slider_x, rp1_slider_y, rp1_slider_w, rp1_slider_h], facecolor="salmon")
            self._slider_rp1 = w.Slider(self._ax_rp1, "$R_P$", r_min, r_max, valinit=r_min, valfmt='%.2f')
            self._ax_rp1.set_title("Planet position", fontsize='small')
            self._slider_rp1.on_changed(self.update)

            # planet mass in units of log10(M_star)
            self._ax_mp1 = self.fig.add_axes([mp1_slider_x, mp1_slider_y, mp1_slider_w, mp1_slider_h], facecolor="salmon")
            self._slider_mp1 = w.Slider(self._ax_mp1, "$M_P$", -4, -2, valinit=-4, valfmt='%.2f')
            self._ax_mp1.set_title("Planet Mass", fontsize='small')
            self._slider_mp1.on_changed(self.update)

            self.marker_planet_dist1, = self.ax.loglog(self.r[100] / au, self.sigma[-1, 0], marker='o', linestyle='None', markersize=10, color="salmon", markeredgecolor='black')
            """self.R1_marker1, = self.ax.loglog(self.r[0] / au, self.sigma[-1,0], marker="x", linestyle='None', markersize=10)
            self.R2_marker1, = self.ax.loglog(self.r[0] / au, self.sigma[-1,0], marker="x", linestyle='None', markersize=10)"""

        # Planet slider 2
        if self.planet[1] == 1:
            self._ax_rp2 = self.fig.add_axes([rp2_slider_x, rp2_slider_y, rp2_slider_w, rp2_slider_h], facecolor="turquoise")
            self._slider_rp2 = w.Slider(self._ax_rp2, "$R_P$", r_min, r_max, valinit=r_min, valfmt='%.2f')
            self._ax_rp2.set_title("$position: $", fontsize='small')
            self._slider_rp2.on_changed(self.update)

            self._ax_mp2 = self.fig.add_axes([mp2_slider_x, mp2_slider_y, mp2_slider_w, mp2_slider_h], facecolor="turquoise")
            self._slider_mp2 = w.Slider(self._ax_mp2, "$M_P$", -4, -2, valinit=-4, valfmt='%.2f')
            self._ax_mp2.set_title("Planet mass", fontsize='small')
            self._slider_mp2.on_changed(self.update)

            self.marker_planet_dist2, = self.ax.loglog(self.r[0] / au, self.sigma[-1, 0], marker='o', linestyle='None', markersize=10, color="turquoise", markeredgecolor='black')

        # Planet sliders 3
        if self.planet[2] == 1:
            self._ax_rp3 = self.fig.add_axes([rp3_slider_x, rp3_slider_y, rp3_slider_w, rp3_slider_h], facecolor="khaki")
            self._slider_rp3 = w.Slider(self._ax_rp3, "$R_P$", r_min, r_max, valinit=r_min, valfmt='%.2f')
            self._ax_rp3.set_title("planet position", fontsize='small')
            self._slider_rp3.on_changed(self.update)

            self._ax_mp3 = self.fig.add_axes([mp3_slider_x, mp3_slider_y, mp3_slider_w, mp3_slider_h], facecolor="khaki")
            self._slider_mp3 = w.Slider(self._ax_mp3, "$M_P$", -4, -2, valinit=-4, valfmt='%.2f')
            self._ax_mp3.set_title("planet mass", fontsize='small')
            self._slider_mp3.on_changed(self.update)

            self.marker_planet_dist3, = self.ax.loglog(self.r[0] / au, self.sigma[-1, 0], marker='o', linestyle='None', markersize=10, color="khaki", markeredgecolor='black')

        # call the callback function once to make the plot agree with state of the buttons

        self.update(None)

    # calculate disturbance in sigma caused by planet in units of Sigma_0
    def f(self, mass_rel, R_p):

        temp      = 20 * (R_p / (100 * au))**-self.q
        cs_p      = np.sqrt(k_b * temp / (mu * m_p))
        h_p       = cs_p / np.sqrt(G * self.M_star / R_p**3)
        K_prime   = (mass_rel)**2 * (h_p / R_p)**-3 * 1 / (10**self._slider_alpha.val)
        K         = K_prime / ((h_p / R_p)**2)
        sig_min_0 = 1 / (1 + 0.04 * K)
        R1        = (sig_min_0 / 4 + 0.08) * K_prime**(1 / 4) * R_p
        R2        = 0.33 * K_prime**(1 / 4) * R_p

        # return R1 and R2 in cgs units
        return R1, R2

    # multiply power law with distubance of planets

    def get_model(self):

        # the widget side

        R_p = []
        mass_ratios = []
        alpha = 10**self._slider_alpha.val
        p = self._slider_Pow.val
        sig0 = 10**(self._slider_sigma.val)

        if self.planet[0] == 1:
            R_p += [10**self._slider_rp1.val * au]
            mass_ratios += [10**self._slider_mp1.val]

        if self.planet[1] == 1:
            R_p += [10**self._slider_rp2.val * au]
            mass_ratios += [10**self._slider_mp2.val]

        if self.planet[2] == 1:
            R_p += [10**self._slider_rp3.val * au]
            mass_ratios += [10**self._slider_mp3.val]

        #self.h_p = np.interp(R_p, self.r, get_disk_height(self.r))
        self.h_p = get_disk_height(np.array(R_p))

        # the model side

        sig_gas = get_surface_density(self.r, alpha, sig0, p, R_p, self.h_p, mass_ratios)

        return sig_gas['sigma']

    def update(self, event):
        """
        The callback for updating the figure when the sliders are moved
        """

        # calculate our toy model
        model_A = self.get_model()
        time_model = self.sigma[:, int(self._slider_T.val)]
        self.alpha = 10.**self._slider_alpha.val

        # MARKERS
        if self.planet[0] == 1:
            """
            # disturbance of 1st planet
            """
            # marker shows x_0 position on model line, label returns analytical function
            """R1, R2 = self.f(10**self._slider_mp1.val, 10**self._slider_rp1.val * au)
            R1 = [10**self._slider_rp1.val + R1 / au, 10**self._slider_rp1.val - R1 / au]
            R2 = [10**self._slider_rp1.val + R2 / au, 10**self._slider_rp1.val - R2 / au]
            self.R1_marker1.set_xdata(R1)
            self.R2_marker1.set_xdata(R2)"""
            self.marker_planet_dist1.set_xdata(10**self._slider_rp1.val)
            self._ax_mp1.set_title("$M = {{{:.2E}}} M_\\odot$".format(10**self._slider_mp1.val, fontsize='small'))
            self._ax_rp1.set_title("$R_P = {{{:.1f}}} AU, h_P = {{{:.2f}}} AU$".format(10**self._slider_rp1.val, self.h_p[0] / au, fontsize='small'))
            self.marker_planet_dist1.set_ydata(np.interp(10**self._slider_rp1.val, self.r / au, model_A))
            """self.R1_marker1.set_ydata(np.interp(R1, self.r / au, model_A))
            self.R2_marker1.set_ydata(np.interp(R2, self.r / au, model_A))"""

        if self.planet[1] == 1:
            """
            # disturbance of 2nd planet
            """
            # marker for 2nd Gaussian and label
            self.marker_planet_dist2.set_xdata(10**self._slider_rp2.val)
            self._ax_mp2.set_title("$M = {{{:.2E}}} M_\\odot$".format(10**self._slider_mp2.val, fontsize='small'))
            self._ax_rp2.set_title("$R_P = {{{:.1f}}} AU, h_P = {{{:.2f}}} AU$".format(10**self._slider_rp2.val, self.h_p[1] / au, fontsize='small'))
            self.marker_planet_dist2.set_ydata(np.interp(10**self._slider_rp2.val, self.r / au, model_A))

        if self.planet[2] == 1:
            """
            # disturbarce of 3rd planet
            """
            # marker and label for 3rd Gaussian
            self.marker_planet_dist3.set_xdata(10**self._slider_rp3.val)
            self.marker_planet_dist3.set_ydata(np.interp(10**self._slider_rp3.val, self.r / au, model_A))
            self._ax_mp3.set_title("$M = {{{:.2E}}} M_\\odot$".format(10**self._slider_mp3.val, fontsize='small'))
            self._ax_rp3.set_title("$R_P = {{{:.1f}}} AU, h_P = {{{:.2f}}} AU$".format(10**self._slider_rp3.val, self.h_p[2] / au, fontsize='small'))

        # update the model line

        self.model_line_T.set_ydata(time_model)
        self.model_line_A.set_ydata(model_A)
        # self.planet_line.set_ydata(self.planet_dist1)

        # update our slider title

        self._ax_T.set_title("t= %.3g a" % (self.t[int(self._slider_T.val)] / (365.25 * 24 * 3600)), fontsize='small')
        self._ax_sigma.set_title("$\\Sigma_0 = 10^{{{:.2g}}} = {{{:.3g}}}$".format(self._slider_sigma.val, 10**self._slider_sigma.val, fontsize='small'))
        self._ax_Pow.set_title("$p = {{{:.2g}}}$".format(self._slider_Pow.val), fontsize='small')
        self._ax_alpha.set_title("$\\alpha = 10^{{{:.2g}}}$".format(self._slider_alpha.val, fontsize='small'))
        #self.model_line_A.set_label(r'$ \Sigma_0 \cdot r^a = {{{:0.4g}}} \cdot (\frac{{r}}{{100AU}})^{{{:.2g}}}$'.format(10**self._slider_sigma.val, self._slider_Pow.val))
        #self.ax.legend(fontsize=13)

        # planet simulations

        plt.draw()

def get_disk_height(R):
    """
    calculate disk height profile from temperature model
    """
    M_star = 2.3*M_sun
    q      = 0.8
    #T      = 20 * (R / (100 * au))**-0.8
    T      = 40.12 * (R / (100 * au))**-0.5
    cs     = np.sqrt(k_b * T / (mu * m_p))
    om     = np.sqrt(G * M_star / R**3)
    h      = cs / om
    return h


def get_surface_density(radius, alpha, sig_0, p, radii, heights, masses):
    """
    Create a power-law surface density profile with multiple planetary gaps.

    Arguments:
    ----------

    radius : array
        radial grid

    alpha : float
        turbulance parameter

    sig_0 : float
        base surface density profile without power law (at r = 1)

    p : float
        power to which radius is taken in surface density profile

    radii : list
        list of planetary positions

    heights : list
        list of pressure scale height at the planet positions

    masses : list
        the mass ratios of planet relative to star for each planet

    Output:
    -------
        surface density profile with planetary gaps in cgs units.
    """
    sig_gas = sig_0 * (radius / (100 * au))**(-p)
    

    if heights is None:
        #heights = np.interp(radii, radius, get_disk_height(radius))
        heights = get_disk_height(np.array(radii))

    for r_p, h, mass in zip(radii, heights, masses):
        kanagawa = kanagawa_profile(radius / r_p, alpha, h / r_p, mass)
        sig_gas *= kanagawa['fact']
        #R1 += [kanagawa['R1']]
        
    R1 = kanagawa['R1']
    
    return {'sigma': sig_gas, 'R1': R1}


def kanagawa_profile(x, alpha, aspect_ratio, mass_ratio, smooth=2.5):
    """Kanagawa planetary gap profile.

    Returns the Kanagawa profile for a planetary gap on the array
    x where x = r / R_p.

    Arguments:
    ----------

    x : array
        radial coordinate in terms of the planet position

    alpha : float
        tubrulence parameter

    aspect_ratio : float
        h / r at the position of the planet

    mass_ratio : float
        planet to star mass ratio

    Output:
    -------
    surface density profile in units of the original surface density.
    """

    K_prime = mass_ratio**2 * aspect_ratio**-3 / alpha
    K = K_prime * (aspect_ratio**-2)
    fact_min_0 = 1 / (1 + 0.04 * K)
    R1 = (fact_min_0 / 4 + 0.08) * K_prime**(1 / 4)
    R2 = 0.33 * K_prime**(1 / 4)
    fact = np.ones_like(x)
    mask = np.abs(x - 1) < R2
    fact[mask] = 4.0 * K_prime**(-1 / 4) * np.abs(x[mask] - 1) - 0.32
    mask = np.abs(x - 1) < R1
    fact[mask] = fact_min_0

    # smoothing
    """
    smooth = 2.5
    x_h    = (mass_ratio / 3.)**(1. / 3.)
    x_s    = smooth * x_h
    fact   = np.exp(np.log(fact) * np.exp(-0.5 * (x - 1)**4 / x_s**4))
    #fact   = np.exp(np.log(fact) * np.exp(-(np.abs(x - 1)/2.3)**3 / R1**4))
    """

    return {'fact': fact, 'R1': ~mask}


def main():

    RTHF = argparse.RawTextHelpFormatter
    PARSER = argparse.ArgumentParser(
        description='Widget to test planetary gap profiles', formatter_class=RTHF)
    PARSER.add_argument('-d', '--data-path',
                        help='path to the data files', type=str, default=None)
    PARSER.add_argument('-n', '--number-planets',
                        help='number of planets to model', type=int, choices=[0, 1, 2, 3])
    PARSER.add_argument('-c', '--choice',
                        help='alternative choice for the data files', type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    ARGS = PARSER.parse_args()

    _ = Widget(data_dir=ARGS.data_path, num_planets=ARGS.number_planets, choose=ARGS.choice)
    plt.show()


if __name__ == '__main__':
    main()
