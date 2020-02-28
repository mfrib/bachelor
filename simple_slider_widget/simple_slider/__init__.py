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

    def __init__(self, data_dir=None, num_planets=None):
        """
        Initialize the widget to compare model and data

        fname : str
            file name to the data file
        """

        # Get the data

        if data_dir is None:
            data_dir = 'data_1_planet'

        # variable to choose number of planets to model
        if num_planets is None:
            num_planets = 1

        self.planet = np.concatenate((np.full(num_planets, 1), np.full(3, 0)))

        # print(num_planets)
        # print(self.planet[0])

        data_dir = pkg_resources.resource_filename(__name__, data_dir)

        self.r = (np.loadtxt(os.path.join(data_dir, 'radius.dat')))
        self.sigma = np.loadtxt(os.path.join(data_dir, 'sigma_averaged.dat'), unpack=1)
        self.t = np.loadtxt(os.path.join(data_dir, 'time.dat'))

        # define disk properties, for now hard-coded

        self.M_star = M_sun
        self.q      = 0.5
        self.T      = 20 * (self.r / (100 * au))**-self.q
        self.cs     = np.sqrt(k_b * self.T / (mu * m_p))
        self.om     = np.sqrt(G * self.M_star / self.r**3)
        self.h      = self.cs / self.om

        # Make the figure

        self.fig = plt.figure(figsize=(12, 12))
        self.ax = self.fig.add_axes([0.1, 0.35, 0.6, 0.6])
        self.ax.set_xlabel(r'r [AU]')
        self.ax.set_ylabel(r'$\Sigma_\mathrm{g}[g/cm^2]$')

        # CREATE SLIDER(S)

        # time slider size

        slider_x0 = self.ax.get_position().x0
        slider_y0 = 0.05
        slider_w = self.ax.get_position().width
        slider_h = 0.04

        # get_model sliders size

        slider_x2 = self.ax.get_position().x0
        slider_y2 = 1 - 0.1 - 19 * slider_h
        slider_w2 = self.ax.get_position().width * 0.4
        slider_h2 = 0.04

        slider_x1 = self.ax.get_position().x0 + self.ax.get_position().width * 0.55
        slider_y1 = 1 - 0.1 - 19 * slider_h
        slider_w1 = self.ax.get_position().width * 0.4
        slider_h1 = 0.04

        # Gaussian sliders size
        # 1st Gaussian, r_0 offset
        g1x_slider_x = 1 - 2.5 * self.ax.get_position().x0
        g1x_slider_y = 1 - 0.1
        g1x_slider_w = self.ax.get_position().width * 0.35
        g1x_slider_h = 0.04

        # 1st Gaussian, width
        g1w_slider_x = g1x_slider_x
        g1w_slider_y = 1 - 0.1 - 2 * g1x_slider_h
        g1w_slider_w = g1x_slider_w
        g1w_slider_h = 0.04

        # 1st Gaussian amplitude
        g1A_slider_x = g1x_slider_x
        g1A_slider_y = 1 - 0.1 - 4 * g1x_slider_h
        g1A_slider_w = g1x_slider_w
        g1A_slider_h = 0.04

        # 2nd Gaussian, r_0 offset
        g2x_slider_x = g1x_slider_x
        g2x_slider_y = 1 - 0.1 - 7.5 * g1x_slider_h
        g2x_slider_w = g1x_slider_w
        g2x_slider_h = 0.04

        # 2nd Gaussian, width
        g2w_slider_x = g1x_slider_x
        g2w_slider_y = 1 - 0.1 - 9.5 * g1x_slider_h
        g2w_slider_w = g1x_slider_w
        g2w_slider_h = 0.04

        # 2nd Gaussian amplitude
        g2A_slider_x = g1x_slider_x
        g2A_slider_y = 1 - 0.1 - 11.5 * g1x_slider_h
        g2A_slider_w = g1x_slider_w
        g2A_slider_h = 0.04

        # 3rd Gaussian, r_0 offset
        g3x_slider_x = g1x_slider_x
        g3x_slider_y = 1 - 0.1 - 15 * g1x_slider_h
        g3x_slider_w = g1x_slider_w
        g3x_slider_h = 0.04

        # 3rd Gaussian, width
        g3w_slider_x = g1x_slider_x
        g3w_slider_y = 1 - 0.1 - 17 * g1x_slider_h
        g3w_slider_w = g1x_slider_w
        g3w_slider_h = 0.04

        # 3rd Gaussian amplitude
        g3A_slider_x = g1x_slider_x
        g3A_slider_y = 1 - 0.1 - 19 * g1x_slider_h
        g3A_slider_w = g1x_slider_w
        g3A_slider_h = 0.04
        """
        #textbox for get_model
        box_x = self.ax.get_position().x0
        box_y = self.ax.get_position().y0 - 2.8*g1x_slider_h
        box_w = self.ax.get_position().width
        box_h = 0.05
        self.axbox = self.fig.add_axes([box_x, box_y, box_w, box_h])
        """
        # useful variable
        r_max = np.log10(self.r[-1] / au)
        r_min = np.log10(self.r[0] / au)

        # slider for parameter T

        self._ax_T = self.fig.add_axes([slider_x0, slider_y0, slider_w, slider_h], facecolor="lightgoldenrodyellow")
        self._slider_T = w.Slider(self._ax_T, "T", 0, 300, valinit=0, valfmt='%.0f')
        self._ax_T.set_title("T", fontsize='small')
        self._slider_T.on_changed(self.update)

        self.model_line_T, = self.ax.loglog(self.r / au, self.sigma[:, int(self._slider_T.val)], 'k-')

        # sliders for get_model

        # power law sliders

        self._ax_Norm = self.fig.add_axes([slider_x2, slider_y2, slider_w2, slider_h2], facecolor="darksalmon")
        self._slider_Norm = w.Slider(self._ax_Norm, "$y_0$", -2, 2, valinit=0, valfmt='%.2f')
        self._ax_Norm.set_title("y_0", fontsize='small')
        self._slider_Norm.on_changed(self.update)

        self._ax_Pow = self.fig.add_axes([slider_x1, slider_y1, slider_w1, slider_h1], facecolor="lightsalmon")
        self._slider_Pow = w.Slider(self._ax_Pow, "a", -2, 0, valinit=-0.8, valfmt='%.2f')
        self._ax_Pow.set_title("Power", fontsize='small')
        self._slider_Pow.on_changed(self.update)

        self.model_line_A, = self.ax.loglog(self.r / au, np.ones_like(self.r))

        # Gaussian sliders 1
        if self.planet[0] == 1:
            self._ax_g1x = self.fig.add_axes([g1x_slider_x, g1x_slider_y, g1x_slider_w, g1x_slider_h], facecolor="salmon")
            self._slider_g1x = w.Slider(self._ax_g1x, "$x_0$", r_min, r_max, valinit=r_min, valfmt='%.2f')
            self._ax_g1x.set_title("Gaussian position", fontsize='small')
            self._slider_g1x.on_changed(self.update)

            self._ax_g1w = self.fig.add_axes([g1w_slider_x, g1w_slider_y, g1w_slider_w, g1w_slider_h], facecolor="salmon")
            self._slider_g1w = w.Slider(self._ax_g1w, "$w$", 0, 1.5, valinit=0, valfmt='%.2f')
            self._ax_g1w.set_title("Gaussian width", fontsize='small')
            self._slider_g1w.on_changed(self.update)

            self._ax_g1A = self.fig.add_axes([g1A_slider_x, g1A_slider_y, g1A_slider_w, g1A_slider_h], facecolor="salmon")
            self._slider_g1A = w.Slider(self._ax_g1A, "$A$", 0, 1, valinit=0, valfmt='%.2f')
            self._ax_g1A.set_title("Gaussian amplitude", fontsize='small')
            self._slider_g1A.on_changed(self.update)

            self.marker_gauss1, = self.ax.loglog(self.r[100] / au, self.sigma[-1, 0], marker='o', linestyle='None', markersize=10, color="salmon", markeredgecolor='black')

        # Gaussian slider 2
        if self.planet[1] == 1:
            self._ax_g2x = self.fig.add_axes(
                [g2x_slider_x, g2x_slider_y, g2x_slider_w, g2x_slider_h], facecolor="turquoise")
            self._slider_g2x = w.Slider(
                self._ax_g2x, "$x_0$", r_min, r_max, valinit=r_min, valfmt='%.2f')
            self._ax_g2x.set_title("Gaussian position", fontsize='small')
            self._slider_g2x.on_changed(self.update)

            self._ax_g2w = self.fig.add_axes(
                [g2w_slider_x, g2w_slider_y, g2w_slider_w, g2w_slider_h], facecolor="turquoise")
            self._slider_g2w = w.Slider(
                self._ax_g2w, "$w$", 0, 1.5, valinit=0, valfmt='%.2f')
            self._ax_g2w.set_title("Gaussian width", fontsize='small')
            self._slider_g2w.on_changed(self.update)

            self._ax_g2A = self.fig.add_axes(
                [g2A_slider_x, g2A_slider_y, g2A_slider_w, g2A_slider_h], facecolor="turquoise")
            self._slider_g2A = w.Slider(
                self._ax_g2A, "$A$", 0, 1, valinit=0, valfmt='%.2f')
            self._ax_g2A.set_title("Gaussian amplitude", fontsize='small')
            self._slider_g2A.on_changed(self.update)
            self.marker_gauss2, = self.ax.loglog(
                self.r[0] / au, self.sigma[-1, 0], marker='o', linestyle='None', markersize=10, color="turquoise", markeredgecolor='black')

        # Gaussian sliders 3
        if self.planet[2] == 1:
            self._ax_g3x = self.fig.add_axes(
                [g3x_slider_x, g3x_slider_y, g3x_slider_w, g3x_slider_h], facecolor="khaki")
            self._slider_g3x = w.Slider(
                self._ax_g3x, "$x_0$", r_min, r_max, valinit=r_min, valfmt='%.2f')
            self._ax_g3x.set_title("Gaussian position", fontsize='small')
            self._slider_g3x.on_changed(self.update)

            self._ax_g3w = self.fig.add_axes(
                [g3w_slider_x, g3w_slider_y, g3w_slider_w, g3w_slider_h], facecolor="khaki")
            self._slider_g3w = w.Slider(
                self._ax_g3w, "$w$", 0, 1.5, valinit=0, valfmt='%.2f')
            self._ax_g3w.set_title("Gaussian width", fontsize='small')
            self._slider_g3w.on_changed(self.update)

            self._ax_g3A = self.fig.add_axes(
                [g3A_slider_x, g3A_slider_y, g3A_slider_w, g3A_slider_h], facecolor="khaki")
            self._slider_g3A = w.Slider(
                self._ax_g3A, "$A$", 0, 1, valinit=0, valfmt='%.2f')
            self._ax_g3A.set_title("Gaussian amplitude", fontsize='small')
            self._slider_g3A.on_changed(self.update)
            self.marker_gauss3, = self.ax.loglog(
                self.r[0] / au, self.sigma[-1, 0], marker='o', linestyle='None', markersize=10, color="khaki", markeredgecolor='black')

        """
        gauss1_text = r'${{{:1.2g}}} \\cdot e^{{-\\frac{{1}}{{2}}(\\frac{{x-{{{:.2f}}}}}{{10^{{{:1.2f}}} }})^2}}$'.format(self._slider_g1A.val, self._slider_g1x.val, self._slider_g1w.val)
        gauss2_text = r'${{{:1.2g}}} \\cdot e^{{-\\frac{{1}}{{2}}(\\frac{{x-{{{:.2f}}}}}{{10^{{{:1.2f}}} }})^2}}$'.format(self._slider_g2A.val, self._slider_g2x.val, self._slider_g2w.val)
        gauss3_text = r'${{{:1.2g}}} \\cdot e^{{-\\frac{{1}}{{2}}(\\frac{{x-{{{:.2f}}}}}{{10^{{{:1.2f}}} }})^2}}$'.format(self._slider_g3A.val, self._slider_g3x.val, self._slider_g3w.val)
        sig_gas_text = r'$10^{{{:.2f}}} \cdot r^{{{:.2f}}}$'.format(self._slider_Norm.val, self._slider_Pow.val)
        self.ax.set_title(sig_gas_text+"$\\cdot (1 - $"+gauss1_text+"$-$"+gauss2_text+"$-$"+gauss3_text+"$)$")
        """

        # call the callback function once to make the plot agree with state of the buttons
        self.update(None)

    def get_model(self):
        """
        # Return analytical function in a textbox
        gauss1_text, gauss2_text, gauss3_text = 0, 0, 0
        if self.planet[0] == 1:
            gauss1_text = '${{{:1.2g}}} \\cdot e^{{-\\frac{{1}}{{2}}(\\frac{{x-{{{:.2f}}}}}{{10^{{{:1.2f}}} }})^2}}$'.format(self._slider_g1A.val, self._slider_g1x.val, self._slider_g1w.val)
        if self.planet[1] == 1:
            gauss2_text = '${{{:1.2g}}} \\cdot e^{{-\\frac{{1}}{{2}}(\\frac{{x-{{{:.2f}}}}}{{10^{{{:1.2f}}} }})^2}}$'.format(self._slider_g2A.val, self._slider_g2x.val, self._slider_g2w.val)
        if self.planet[2] == 1:
            gauss3_text = '${{{:1.2g}}} \\cdot e^{{-\\frac{{1}}{{2}}(\\frac{{x-{{{:.2f}}}}}{{10^{{{:1.2f}}} }})^2}}$'.format(self._slider_g3A.val, self._slider_g3x.val, self._slider_g3w.val)

        sig_gas_text = r'$10^{{{:.2f}}} \cdot r^{{{:.2f}}}$'.format(self._slider_Norm.val, self._slider_Pow.val)
        #self.ax.set_title(sig_gas_text+"$\\cdot (1 - $"+gauss1_text+"$-$"+gauss2_text+"$-$"+gauss3_text+"$)$")
        w.TextBox(self.axbox, '$model$', initial=sig_gas_text+"$\\cdot (1 - $"+gauss1_text+"-"+gauss2_text+"-"+gauss3_text+")")
        """
        sig_gas = 10**(self._slider_Norm.val) * \
            (self.r / (100 * au))**(self._slider_Pow.val)
        self.gauss1, self.gauss2, self.gauss3 = 0, 0, 0
        if self.planet[0] == 1:
            self.gauss1 = (self._slider_g1A.val) * np.exp(-((self.r) - 10**(self._slider_g1x.val) * au)**2 / (2 * (10**(self._slider_g1w.val) * au)**2))

        if self.planet[1] == 1:
            self.gauss2 = (self._slider_g2A.val) * np.exp(-((self.r) - 10**(self._slider_g2x.val) * au)**2 / (2 * (10**(self._slider_g2w.val) * au)**2))

        if self.planet[2] == 1:
            self.gauss3 = (self._slider_g3A.val) * np.exp(-((self.r) - 10**(self._slider_g3x.val) * au)**2 / (2 * (10**(self._slider_g3w.val) * au)**2))
        return sig_gas * (1 - self.gauss1 - self.gauss2 - self.gauss3)

    def update(self, event):
        """
        The callback for updating the figure when the sliders are moved
        """

        # calculate our toy model
        model_A = self.get_model()
        time_model = self.sigma[:, int(self._slider_T.val)]
        # model_A_0 = 10**(self._slider_Norm.val)*self.r**(self._slider_Pow.val)

        # gauss1, gauss2, gauss3 = 0, 0, 0
        # MARKERS
        if self.planet[0] == 1:
            """
            # Gaussian disturbance of 1st planet
            gauss1 = (self._slider_g1A.val)*np.exp(-((self.r)-10**(self._slider_g1x.val))**2 / (2*(10**(self._slider_g1w.val))**2))
            """
            # marker shows x_0 position on model line w/o gaussian amplitude, label returns analytical Gaussian function
            self.marker_gauss1.set_label('$A_1 \\cdot e^{{-\\frac{{1}}{{2}}(\\frac{{r-r_0}}{{w_0}})^2}} = {{{:1.2g}}} \\cdot e^{{-\\frac{{1}}{{2}}(\\frac{{r-{{{:.2f}}}}}{{{{{:1.2g}}} }})^2}}$'.format(self._slider_g1A.val, self._slider_g1x.val, 10**self._slider_g1w.val))
            self.marker_gauss1.set_xdata(10**self._slider_g1x.val)
            self.marker_gauss1.set_ydata(np.interp(10**self._slider_g1x.val, self.r / au, model_A))

        if self.planet[1] == 1:
            """
            # Gaussian disturbance of 2nd planet
            gauss2 = (self._slider_g2A.val)*np.exp(-((self.r)-10**(self._slider_g2x.val))**2 / (2*(10**(self._slider_g2w.val))**2))
            """
            # marker for 2nd Gaussian and label
            self.marker_gauss2.set_label('$A_2 \\cdot e^{{-\\frac{{1}}{{2}}(\\frac{{r-r_0}}{{w_0}})^2}} = {{{:1.2g}}} \\cdot e^{{-\\frac{{1}}{{2}}(\\frac{{r-{{{:.2f}}}}}{{{{{:1.2g}}} }})^2}}$'.format(
                self._slider_g2A.val, self._slider_g2x.val, 10**self._slider_g2w.val))
            self.marker_gauss2.set_xdata(10**self._slider_g2x.val)
            self.marker_gauss2.set_ydata(np.interp(10**self._slider_g2x.val, self.r / au, model_A))

        if self.planet[2] == 1:
            """
            # Gaussian disturbarce of 3rd planet
            gauss3 = (self._slider_g3A.val)*np.exp(-((self.r)-10**(self._slider_g3x.val))**2 / (2*(10**(self._slider_g3w.val))**2))
            """
            # marker and label for 3rd Gaussian
            self.marker_gauss3.set_label('$A_3 \\cdot e^{{-\\frac{{1}}{{2}}(\\frac{{r-r_0}}{{w_0}})^2}} = {{{:1.2g}}} \\cdot e^{{-\\frac{{1}}{{2}}(\\frac{{r-{{{:.2f}}}}}{{{{{:1.2g}}} }})^2}}$'.format(self._slider_g3A.val, self._slider_g3x.val, 10**self._slider_g3w.val))
            self.marker_gauss3.set_xdata(10**self._slider_g3x.val)
            self.marker_gauss3.set_ydata(np.interp(10**self._slider_g3x.val, self.r / au, model_A))

        # model_A = model_A_0 * (1 - gauss1 - gauss2 - gauss3)

        # update the model line

        self.model_line_T.set_ydata(time_model)
        self.model_line_A.set_ydata(model_A)

        # update our slider title

        self._ax_T.set_title("t= %.3g a" % (
            self.t[int(self._slider_T.val)] / (365.25 * 24 * 3600)), fontsize='small')
        self._ax_Norm.set_title("$y_0 = 10^{{{:.2f}}}$".format(
            self._slider_Norm.val, fontsize='small'))
        self._ax_Pow.set_title("$a = {{{:.2f}}}$".format(
            self._slider_Pow.val), fontsize='small')
        self.model_line_A.set_label(r'$ y_0 \cdot r^a = {{{:0.4g}}} \cdot r^{{{:.2g}}}$'.format(
            10**self._slider_Norm.val, self._slider_Pow.val))
        self.ax.legend(fontsize=15)

        # planet simulations

        plt.draw()


def main():

    RTHF = argparse.RawTextHelpFormatter
    PARSER = argparse.ArgumentParser(
        description='Widget to test planetary gap profiles', formatter_class=RTHF)
    PARSER.add_argument('-d', '--data-path',
                        help='path to the data files', type=str, default=None)
    PARSER.add_argument('-n', '--number-planets',
                        help='number of planets to model', type=int, choices=[0, 1, 2, 3])
    ARGS = PARSER.parse_args()

    _ = Widget(data_dir=ARGS.data_path, num_planets=ARGS.number_planets)
    plt.show()


if __name__ == '__main__':
    main()
