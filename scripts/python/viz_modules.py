"""Visualization module
"""
import numpy as np
import matplotlib.pyplot as plt
from sim_modules import ExperimentData

def plot_heatmap():
    pass

def plot_timeseries(target_fill_ratio, sensor_prob, data_obj: ExperimentData):
    """Plot the time series data.

    Create data visualization for local, social, and informed values for a simulation with a
    specific target fill ratio and sensor probability.

    Args:
        target_fill_ratio: The target fill ratio used in the simulation data.
        sensor_prob: The sensor probability used in the simulation data.
        data_obj: A ExperimentData object containing all the simulation data.
    """

    # Create figure and axes handles
    fig_x_hat, ax_x_hat = plt.subplots(2)
    fig_x_hat.set_size_inches(8,6)

    fig_x_bar, ax_x_bar = plt.subplots(2)
    fig_x_bar.set_size_inches(8,6)

    fig_x, ax_x = plt.subplots(2)
    fig_x.set_size_inches(8,6)

    abscissa_values_x_hat = list(range(data_obj.num_obs))
    abscissa_values_x_bar = list(range(0, data_obj.num_obs, data_obj.comms_period))
    abscissa_values_x = list(range(0, data_obj.num_obs, data_obj.comms_period))

    # Plot for all experiments
    for n in range(data_obj.num_exp):
        sim_obj = data_obj.get_stats_obj(target_fill_ratio, sensor_prob)

        # Plot time evolution of local estimates and confidences
        x_hat_bounds = compute_std_bounds(sim_obj.x_hat_sample_mean[n], sim_obj.x_hat_sample_std[n])
        alpha_bounds = compute_std_bounds(sim_obj.alpha_sample_mean[n], sim_obj.alpha_sample_std[n])

        ax_x_hat[0].plot(abscissa_values_x_hat, sim_obj.x_hat_sample_mean[n], label="Exp {}".format(n))
        ax_x_hat[0].fill_between(abscissa_values_x_hat, x_hat_bounds[0], x_hat_bounds[1], alpha=0.2)

        ax_x_hat[1].plot(sim_obj.alpha_sample_mean[n], label="Exp {}".format(n))
        ax_x_hat[1].fill_between(abscissa_values_x_hat, alpha_bounds[0], alpha_bounds[1], alpha=0.2)

        # Plot time evolution of social estimates and confidences
        x_bar_bounds = compute_std_bounds(sim_obj.x_bar_sample_mean[n], sim_obj.x_bar_sample_std[n])
        rho_bounds = compute_std_bounds(sim_obj.rho_sample_mean[n], sim_obj.rho_sample_std[n])

        ax_x_bar[0].plot(abscissa_values_x_bar, sim_obj.x_bar_sample_mean[n], label="Exp {}".format(n))
        ax_x_bar[0].fill_between(abscissa_values_x_bar, x_bar_bounds[0], x_bar_bounds[1], alpha=0.2)

        ax_x_bar[1].plot(abscissa_values_x_bar, sim_obj.rho_sample_mean[n], label="Exp {}".format(n))
        ax_x_bar[1].fill_between(abscissa_values_x_bar, rho_bounds[0], rho_bounds[1], alpha=0.2)

        # Plot time evolution of informed estimates and confidences
        x_bounds = compute_std_bounds(sim_obj.x_sample_mean[n], sim_obj.x_sample_std[n])
        gamma_bounds = compute_std_bounds(sim_obj.gamma_sample_mean[n], sim_obj.gamma_sample_std[n])

        ax_x[0].plot(abscissa_values_x, sim_obj.x_sample_mean[n], label="Exp {}".format(n))
        ax_x[0].fill_between(abscissa_values_x, x_bounds[0], x_bounds[1], alpha=0.2)

        ax_x[1].plot(abscissa_values_x, sim_obj.gamma_sample_mean[n], label="Exp {}".format(n))
        ax_x[1].fill_between(abscissa_values_x, gamma_bounds[0], gamma_bounds[1], alpha=0.2)

    # Set axis properties
    ax_x_hat[0].set_title("Average of {0} agents' local values with 1\u03c3 bounds (fill ratio: {1}, sensor: {2})".format(data_obj.num_agents, target_fill_ratio, sensor_prob))
    ax_x_hat[0].set_ylabel("Local estimates")
    ax_x_hat[0].set_ylim(0, 1.0)
    ax_x_hat[1].set_ylabel("Local confidences")
    ax_x_hat[1].set_xlabel("Observations")
    ax_x_hat[1].set_yscale("log")

    ax_x_bar[0].set_title("Average of {0} agents' social values with 1\u03c3 bounds (fill ratio: {1}, sensor: {2})".format(data_obj.num_agents, target_fill_ratio, sensor_prob))
    ax_x_bar[0].set_ylabel("Social estimates")
    ax_x_bar[0].set_ylim(0, 1.0)
    ax_x_bar[1].set_ylabel("Social confidences")
    ax_x_bar[1].set_xlabel("Observations")
    ax_x_bar[1].set_yscale("log")

    ax_x[0].set_title("Average of {0} agents' informed values with 1\u03c3 bounds (fill ratio: {1}, sensor: {2})".format(data_obj.num_agents, target_fill_ratio, sensor_prob))
    ax_x[0].set_ylabel("Informed estimates")
    ax_x[0].set_ylim(0, 1.0)
    ax_x[1].set_ylabel("Informed confidences")
    ax_x[1].set_xlabel("Observations")
    ax_x[1].set_yscale("log")

    # Turn on grid lines
    activate_subplot_grid_lines(ax_x_hat)
    activate_subplot_grid_lines(ax_x_bar)
    activate_subplot_grid_lines(ax_x)

    # Adjust legend location and plot sizes
    adjust_subplot_legend_and_axis(fig_x_hat, ax_x_hat)
    adjust_subplot_legend_and_axis(fig_x_bar, ax_x_bar)
    adjust_subplot_legend_and_axis(fig_x, ax_x)

    plt.show()

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