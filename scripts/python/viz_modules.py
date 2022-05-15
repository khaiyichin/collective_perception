"""Visualization module
"""
import numpy as np
import matplotlib.pyplot as plt
from sim_modules import ExperimentData

class VisualizationData:
    """Class for storing simulation results for easy data analysis and visualization.

    The ExperimentData class is not used for visualization (despite providing similar functionality)
    because of its RAM footprint (since it stores agent observations and tile configurations, none
    of which are needed in data visualization).

    Furthermore, due to the size of simulations, multiple runs are needed to cover the desired
    simulation parameters. This means that multiple ExperimentData objects have to be stored and
    loaded for data visualization, which complicates the structure of the ExperimentData class if
    we were to also use it for data visualization. Thus it's easier to just create a separate class
    that takes in the paths to the multiple ExperimentData to extract the data.
    """

    class AggregateStats:

        def __init__(self):

            self.x_hat_mean = []
            self.x_bar_mean = []
            self.x_mean = []

            self.alpha_mean = []
            self.rho_mean = []
            self.gamma_mean = []

            self.x_hat_std = []
            self.x_bar_std = []
            self.x_std = []

            self.alpha_std = []
            self.rho_std = []
            self.gamma_std = []

            self.x_hat_conv_ind = -1
            self.x_bar_conv_ind = -1
            self.x_conv_ind = -1

    def __init__(self, exp_data_obj_paths, difference_window_size=7, convergence_thresh=5e-3):

        # Load experiment data statistics
        first_obj = True

        self.difference_window_size = difference_window_size
        self.convergence_thresh = convergence_thresh

        for exp_data_obj_path in exp_data_obj_paths:
            obj = ExperimentData.load(exp_data_obj_path, False)

            if first_obj:
                self.num_agents = obj.num_agents
                self.num_exp = obj.num_exp
                self.num_obs = obj.num_obs
                self.graph_type = obj.graph_type
                self.comms_period = obj.comms_period
                self.comms_prob = obj.comms_prob
                self.dfr_range = obj.dfr_range
                self.sp_range = obj.sp_range
                self.stats_obj_dict = obj.stats_obj_dict
                
                first_obj = False
            else:
                similarity_bool = True
                similarity_bool = similarity_bool and (self.num_agents == obj.num_agents)
                similarity_bool = similarity_bool and (self.num_exp == obj.num_exp)
                similarity_bool = similarity_bool and (self.num_obs == obj.num_obs)
                similarity_bool = similarity_bool and (self.graph_type == obj.graph_type)
                similarity_bool = similarity_bool and (self.comms_period == obj.comms_period)
                similarity_bool = similarity_bool and (self.comms_prob == obj.comms_prob)
                similarity_bool = similarity_bool and (self.sp_range == obj.sp_range)

                # Only the target fill ratio range can be different (highest level)
                if not similarity_bool:
                    raise Exception("The ExperimentData objects do not have the same parameters.")
                else:
                    self.dfr_range.extend(obj.dfr_range)
                    self.stats_obj_dict.update(obj.stats_obj_dict)

        self.aggregate_statistics()

    def aggregate_statistics(self):
        """Aggregate the statistics.

        The estimates (and corresponding std. devs.) from the experiments within each simulation parameter setting
        will be combined to form one mean estimate (std. devs).
        """

        self.agg_stats_dict = {}

        for dfr_key, sp_dict in self.stats_obj_dict.items():

            temp_dict = {}

            for sp_key, stats_obj in sp_dict.items():

                # Compute mean across all experiments
                x_hat_mean = np.mean(stats_obj.x_hat_sample_mean, axis=0)
                x_bar_mean = np.mean(stats_obj.x_bar_sample_mean, axis=0)
                x_mean = np.mean(stats_obj.x_sample_mean, axis=0)

                alpha_mean = np.mean(stats_obj.alpha_sample_mean, axis=0)
                rho_mean = np.mean(stats_obj.rho_sample_mean, axis=0)
                gamma_mean = np.mean(stats_obj.gamma_sample_mean, axis=0)

                # Compute pooled variance across all experiments
                x_hat_std = np.sqrt( np.mean( np.square( stats_obj.x_hat_sample_std ), axis=0) )
                x_bar_std = np.sqrt( np.mean( np.square( stats_obj.x_bar_sample_std ), axis=0) )
                x_std = np.sqrt( np.mean( np.square( stats_obj.x_sample_std ), axis=0) )

                alpha_std = np.sqrt( np.mean( np.square( stats_obj.alpha_sample_std ), axis=0) )
                rho_std = np.sqrt( np.mean( np.square( stats_obj.rho_sample_std ), axis=0) )
                gamma_std = np.sqrt( np.mean( np.square( stats_obj.gamma_sample_std ), axis=0) )

                # Compute convergence for the aggregate estimates
                x_hat_conv_ind = self.detect_convergence(x_hat_mean,
                                                         self.difference_window_size*self.comms_period,
                                                         self.convergence_thresh)
                x_bar_conv_ind = self.detect_convergence(x_bar_mean,
                                                         self.difference_window_size,
                                                         self.convergence_thresh)
                x_conv_ind = self.detect_convergence(x_mean,
                                                     self.difference_window_size,
                                                     self.convergence_thresh)

                # Store values into dictionary
                agg_stats_obj = self.AggregateStats()

                agg_stats_obj.x_hat_mean = x_hat_mean
                agg_stats_obj.x_bar_mean = x_bar_mean
                agg_stats_obj.x_mean = x_mean
                agg_stats_obj.alpha_mean = alpha_mean
                agg_stats_obj.rho_mean = rho_mean
                agg_stats_obj.gamma_mean = gamma_mean
                agg_stats_obj.x_hat_std = x_hat_std
                agg_stats_obj.x_bar_std = x_bar_std
                agg_stats_obj.x_std = x_std
                agg_stats_obj.alpha_std = alpha_std
                agg_stats_obj.rho_std = rho_std
                agg_stats_obj.gamma_std = gamma_std
                agg_stats_obj.x_hat_conv_ind = x_hat_conv_ind
                agg_stats_obj.x_bar_conv_ind = x_bar_conv_ind
                agg_stats_obj.x_conv_ind = x_conv_ind

                temp_dict[sp_key] = agg_stats_obj

            self.agg_stats_dict[dfr_key] = temp_dict

    def detect_convergence(self, curve, window_size, threshold):
        """Compute the point in time when convergence is achieved.

        Args:
            curve: A numpy array of points.
            window_size: An integer parametrizing the window size for convergence evaluation.
            threshold: A float parametrizing the difference threshold.

        Returns:
            The index at which convergence criterion is achieved.
        """

        # Compute the difference curve
        difference_curve = self.compute_difference(curve, window_size)

        # Find the convergence index
        indices = np.argwhere(abs(difference_curve) <= threshold).T[0]
        candidate = -window_size # arbitrary negative number to ensure that if convergence criterion isn't met then output is negative

        for ind, val in enumerate(indices):

            # Assign candidate value
            if candidate == -window_size:
                candidate = val

            # Verify that the candidate has converged
            if (ind+1 != len(indices)):
                if (val+1 == indices[ind+1]):
                    continue
                else:
                    candidate = -window_size

        conv_ind = candidate + window_size//2 # TODO: need to verify that the index is indeed correct (central, or first occurrence?)

        return conv_ind

    def compute_difference(self, curve, window_size):
        """Compute the difference between endpoints the length of a specified window size.

        The difference is computed using a two-sided method, i.e., central difference.

        Args:
            curve: A numpy array of points.
            window_size: An integer parametrizing the window size for central difference calculation.
        """

        assert(window_size%2 != 0) # window size must be odd

        dx = []

        side_len = window_size//2

        for ind in range(len(curve)):

            start_ind = ind - side_len
            end_ind = ind + side_len

            if (start_ind < 0): continue
            elif (end_ind >= len(curve)): break
            else:
                dx.append( curve[start_ind] - curve[end_ind] )

        assert( len(dx) == (len(curve) - window_size + 1) ) # ensure that the output list has the correct length

        return np.asarray(dx)

def plot_heatmap():
    pass

def plot_timeseries(target_fill_ratio, sensor_prob, data_obj: VisualizationData, agg_data=False):
    """Plot the time series data.

    Create data visualization for local, social, and informed values for a simulation with a
    specific target fill ratio and sensor probability.

    Args:
        target_fill_ratio: The target fill ratio used in the simulation data.
        sensor_prob: The sensor probability used in the simulation data.
        data_obj: A VisualizationData object containing all the simulation data.
        agg_data: Boolean to use aggregate data.
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
    if not agg_data:
        for n in range(data_obj.num_exp):
            stats_obj = data_obj.stats_obj_dict[target_fill_ratio][sensor_prob]

            # Plot time evolution of local estimates and confidences
            x_hat_bounds = compute_std_bounds(stats_obj.x_hat_sample_mean[n], stats_obj.x_hat_sample_std[n])
            alpha_bounds = compute_std_bounds(stats_obj.alpha_sample_mean[n], stats_obj.alpha_sample_std[n])

            ax_x_hat[0].plot(abscissa_values_x_hat, stats_obj.x_hat_sample_mean[n], label="Exp {}".format(n))
            ax_x_hat[0].fill_between(abscissa_values_x_hat, x_hat_bounds[0], x_hat_bounds[1], alpha=0.2)

            ax_x_hat[1].plot(stats_obj.alpha_sample_mean[n], label="Exp {}".format(n))
            ax_x_hat[1].fill_between(abscissa_values_x_hat, alpha_bounds[0], alpha_bounds[1], alpha=0.2)

            # Plot time evolution of social estimates and confidences
            x_bar_bounds = compute_std_bounds(stats_obj.x_bar_sample_mean[n], stats_obj.x_bar_sample_std[n])
            rho_bounds = compute_std_bounds(stats_obj.rho_sample_mean[n], stats_obj.rho_sample_std[n])

            ax_x_bar[0].plot(abscissa_values_x_bar, stats_obj.x_bar_sample_mean[n], label="Exp {}".format(n))
            ax_x_bar[0].fill_between(abscissa_values_x_bar, x_bar_bounds[0], x_bar_bounds[1], alpha=0.2)

            ax_x_bar[1].plot(abscissa_values_x_bar, stats_obj.rho_sample_mean[n], label="Exp {}".format(n))
            ax_x_bar[1].fill_between(abscissa_values_x_bar, rho_bounds[0], rho_bounds[1], alpha=0.2)

            # Plot time evolution of informed estimates and confidences
            x_bounds = compute_std_bounds(stats_obj.x_sample_mean[n], stats_obj.x_sample_std[n])
            gamma_bounds = compute_std_bounds(stats_obj.gamma_sample_mean[n], stats_obj.gamma_sample_std[n])

            ax_x[0].plot(abscissa_values_x, stats_obj.x_sample_mean[n], label="Exp {}".format(n))
            ax_x[0].fill_between(abscissa_values_x, x_bounds[0], x_bounds[1], alpha=0.2)

            ax_x[1].plot(abscissa_values_x, stats_obj.gamma_sample_mean[n], label="Exp {}".format(n))
            ax_x[1].fill_between(abscissa_values_x, gamma_bounds[0], gamma_bounds[1], alpha=0.2)

    else:
        agg_stats_obj = data_obj.agg_stats_dict[target_fill_ratio][sensor_prob]

        # Plot time evolution of local estimates and confidences
        x_hat_bounds = compute_std_bounds(agg_stats_obj.x_hat_mean, agg_stats_obj.x_hat_std)
        alpha_bounds = compute_std_bounds(agg_stats_obj.alpha_mean, agg_stats_obj.alpha_std)

        ax_x_hat[0].plot(abscissa_values_x_hat, agg_stats_obj.x_hat_mean)
        ax_x_hat[0].axvline(abscissa_values_x_hat[agg_stats_obj.x_hat_conv_ind], color="black", linestyle=":")
        ax_x_hat[0].axhline(target_fill_ratio, color="black", linestyle="--")
        ax_x_hat[0].fill_between(abscissa_values_x_hat, x_hat_bounds[0], x_hat_bounds[1], alpha=0.2)

        ax_x_hat[1].plot(agg_stats_obj.alpha_mean)
        ax_x_hat[1].axvline(abscissa_values_x_hat[agg_stats_obj.x_hat_conv_ind], color="black", linestyle=":")
        ax_x_hat[1].fill_between(abscissa_values_x_hat, alpha_bounds[0], alpha_bounds[1], alpha=0.2)

        # Plot time evolution of social estimates and confidences
        x_bar_bounds = compute_std_bounds(agg_stats_obj.x_bar_mean, agg_stats_obj.x_bar_std)
        rho_bounds = compute_std_bounds(agg_stats_obj.rho_mean, agg_stats_obj.rho_std)

        ax_x_bar[0].plot(abscissa_values_x_bar, agg_stats_obj.x_bar_mean)
        ax_x_bar[0].axvline(abscissa_values_x_bar[agg_stats_obj.x_bar_conv_ind], color="black", linestyle=":")
        ax_x_bar[0].axhline(target_fill_ratio, color="black", linestyle="--")
        ax_x_bar[0].fill_between(abscissa_values_x_bar, x_bar_bounds[0], x_bar_bounds[1], alpha=0.2)

        ax_x_bar[1].plot(abscissa_values_x_bar, agg_stats_obj.rho_mean)
        ax_x_bar[1].axvline(abscissa_values_x_bar[agg_stats_obj.x_bar_conv_ind], color="black", linestyle=":")
        ax_x_bar[1].fill_between(abscissa_values_x_bar, rho_bounds[0], rho_bounds[1], alpha=0.2)

        # Plot time evolution of informed estimates and confidences
        x_bounds = compute_std_bounds(agg_stats_obj.x_mean, agg_stats_obj.x_std)
        gamma_bounds = compute_std_bounds(agg_stats_obj.gamma_mean, agg_stats_obj.gamma_std)

        ax_x[0].plot(abscissa_values_x, agg_stats_obj.x_mean)
        ax_x[0].axvline(abscissa_values_x[agg_stats_obj.x_conv_ind], color="black", linestyle=":")
        ax_x[0].axhline(target_fill_ratio, color="black", linestyle="--")
        ax_x[0].fill_between(abscissa_values_x, x_bounds[0], x_bounds[1], alpha=0.2)

        ax_x[1].plot(abscissa_values_x, agg_stats_obj.gamma_mean)
        ax_x[1].axvline(abscissa_values_x[agg_stats_obj.x_conv_ind], color="black", linestyle=":")
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