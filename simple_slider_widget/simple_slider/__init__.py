import matplotlib.pyplot as plt
import matplotlib.widgets as w
import numpy as np
import pkg_resources
import argparse
import os


class Widget():

    def __init__(self, data_dir=None):
        """
        Initialize the widget to compare model and data

        fname : str
            file name to the data file
        """

        # Get the data

        # redundant in this version

        if data_dir is None:
            data_dir = 'data_1_planet'

        data_dir = pkg_resources.resource_filename(__name__, data_dir)

        self.r     = np.loadtxt(os.path.join(data_dir, 'radius.dat'))
        self.sigma = np.loadtxt(os.path.join(data_dir, 'sigma_averaged.dat'), unpack=1)
        self.t     = np.loadtxt(os.path.join(data_dir, 'time.dat'))

        # Make the figure

        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0.1, 0.3, 0.5, 0.6])

        # plot the reference profile

        # CREATE SLIDER(S)

        slider_x0 = self.ax.get_position().x0
        slider_y0 = 0.05
        slider_w  = self.ax.get_position().width
        slider_h  = 0.04

        # slider for parameter T

        self._ax_T = self.fig.add_axes([slider_x0, slider_y0, slider_w, slider_h], facecolor="lightgoldenrodyellow")
        self._slider_T = w.Slider(self._ax_T, "T", 0, 300, valinit=0, valfmt='%.0f')
        self._ax_T.set_title("T", fontsize='small')
        self._slider_T.on_changed(self.update)

        self.model_line, = self.ax.loglog(self.r, self.sigma[:, int(self._slider_T.val)])

        # call the callback function once to make the plot agree with state of the buttons

        self.update(None)

    def update(self, event):
        """
        The callback for updating the figure when the sliders are moved
        """

        # calculate our toy model

        time_model = self.sigma[:, int(self._slider_T.val)]

        # update the model line

        self.model_line.set_ydata(time_model)

        # update our slider title

        self._ax_T.set_title("t= %.3e s" % self.t[int(self._slider_T.val)], fontsize='small')

        plt.draw()


def main():

    RTHF   = argparse.RawTextHelpFormatter
    PARSER = argparse.ArgumentParser(description='Widget to test planetary gap profiles', formatter_class=RTHF)
    PARSER.add_argument('-d', '--data-path', help='path to the data files', type=str, default=None)
    ARGS  = PARSER.parse_args()

    _ = Widget(data_dir=ARGS.data_path)
    plt.show()


if __name__ == '__main__':
    main()
