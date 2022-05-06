"""Visualization module
"""
import numpy as np

def plot_heatmap():
    pass

def plot_timeseries(): # ???
    pass

def compute_std_bounds(mean_val, std_val):
    return [ np.add(mean_val, std_val),  np.subtract(mean_val, std_val) ]

def compute_convergence(trajectory, window_size):
    """Compute the point in time when convergence is achieved.

    The methods used here can be parametrized.
    TODO: for now just using a simple moving average filter
    """

    # Use a simple moving average filter
    pass

def compute_cma(trajectory, window_size):
    """Apply the central moving average filter on a trajectory.
    """

    assert(window_size%2 != 0) # window size must be odd

    return np.convolve(trajectory, np.ones(window_size), 'valid') / window_size

def compute_derivative(a, b, dt=1): return (a-b)/dt

def activate_subplot_grid_lines(subplot_ax):
    for ax in subplot_ax:
        ax.minorticks_on()
        ax.xaxis.grid()
        ax.yaxis.grid()

def adjust_subplot_legend_and_axis(subplot_fig, subplot_ax):
    box_1 = subplot_ax[0].get_position()
    box_2 = subplot_ax[1].get_position()
    subplot_ax[0].set_position([box_1.x0, box_1.y0, box_1.width * 0.9, box_1.height])
    subplot_ax[1].set_position([box_2.x0, box_2.y0, box_2.width * 0.9, box_2.height])

    handles, labels = subplot_ax[1].get_legend_handles_labels()
    subplot_fig.legend(handles, labels, loc='center right', bbox_to_anchor=(box_2.width*1.25, 0.5))