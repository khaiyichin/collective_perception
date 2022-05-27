"""Visualization module
"""
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
import warnings
from joblib import Parallel, delayed

from sim_modules import ExperimentData

# Default values
CONV_THRESH = 5e-3

warnings.filterwarnings("ignore", category=UserWarning) # ignore UserWarning type warnings

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

    This class is intended to store simulations with the same:
        - communication network type,
        - number of experiments,
        - number of agents,
        - communication period, and
        - communication probability
    and with varying:
        - target fill ratios, and
        - sensor probabilities.
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

    def __init__(self, exp_data_obj_folder):

        # Load experiment data statistics
        first_obj = True

        # Iterate through folder contents recursively to obtain all serialized filenames
        exp_data_obj_paths = []

        for root, _, files in os.walk(exp_data_obj_folder):
            for f in files:
                if os.path.splitext(f)[1] == ".pkl": # currently serialized files are pickled
                    exp_data_obj_paths.append( os.path.join(root, f) )

        # Load the files
        for path in exp_data_obj_paths:
            obj = ExperimentData.load(path, False)

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
                self.agg_stats_dict = {} # to be populated later
                
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
                    raise RuntimeError("The ExperimentData objects in the \"{0}\" directory do not have the same parameters.".format(exp_data_obj_paths))
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

                temp_dict[sp_key] = agg_stats_obj

            self.agg_stats_dict[dfr_key] = temp_dict

    def detect_convergence(self, target_fill_ratio: float, sensor_prob: float, threshold=CONV_THRESH):
        """Compute the point in time when convergence is achieved.

        This computes the convergence timestep (# of observations) for the local, social,
        and informed estimates. If the returned value is equal to the number of observations,
        that means convergence was not achieved.

        Args:
            target_fill_ratio: The target fill ratio used in the simulation.
            sensor_prob: The sensor probability used in the simulation.
            threshold: A float parametrizing the difference threshold.

        Returns:
            The 3 indices at which convergence criterion is achieved.
        """

        curves = [
            self.agg_stats_dict[target_fill_ratio][sensor_prob].x_hat_mean, # length of self.num_obs + 1
            self.agg_stats_dict[target_fill_ratio][sensor_prob].x_bar_mean, # length of self.num_obs / self.comms_period + 1
            self.agg_stats_dict[target_fill_ratio][sensor_prob].x_mean, # length of self.num_obs / self.comms_period + 1
        ]

        """
        Methodology in computing convergence:
        Anchor to one point and check all later values to see if difference exceeds threshold,
        repeat until the earliest anchor point reaches the end uninterrupted.
        """

        # Define local function used for processing the inner for loop in parallel
        def parallel_inner_loop(curve):

            # Iterate through each point in the curve as reference
            for ref_ind, ref in enumerate(curve):

                running_ind = ref_ind

                # Compute the difference between the reference point and each of the points following it
                while ( running_ind < len(curve) ) and ( abs(ref - curve[running_ind]) < threshold ): running_ind += 1

                # Return the reference point index
                # True only if convergence is early or the entire curve has been traversed, which means the index of the last point is used
                if running_ind == len(curve): # found it or it's the last point
                    return ref_ind

        # Go through all the curves
        conv_ind = Parallel(n_jobs=3, verbose=0)(delayed(parallel_inner_loop)(c) for c in curves)

        return conv_ind[0], conv_ind[1], conv_ind[2]

class VisualizationDataGroup:
    """Class to store VisualizationData objects.

    The VisualizationData objects are stored by first using the VisualizationData's method
    in loading serialized ExperimentData files. Then each VisualizationData object
    are stored in this class.

    This class is intended to store VisualizationData objects with the same:
        - communication network type, and
        - number of experiments,
    and with varying:
        - number of agents,
        - communication period,
        - communication probability.
    The target fill ratios and sensor probabilities are already varied (with fixed ranges) in the
    stored VisualizationData objects.

    This class is initialized with 1 argument:
        data_folder: A string specifying the directory containing all the ExperimentData files.
    """

    def __init__(self, data_folder):

        self.viz_data_obj_dict = {}
        self.stored_obj_counter = 0

        # Iterate through folder contents recursively to obtain folders containing the serialized files
        exp_data_obj_folders = []

        for root, _, files in os.walk(data_folder):

            # Check to see if any pickle file exist in the current directory
            serialized_files = [f for f in files if os.path.splitext(f)[1] == ".pkl"]

            if len(serialized_files) == 0: # move on to the next folder
                continue
            else: # pickle file found, that means the directory two levels up is needed
                parent_folder, folder = os.path.split( os.path.abspath(root) )
                exp_data_obj_folders.append(parent_folder)

        self.folders = list(set(exp_data_obj_folders)) # store the unique values for the paths

        # Load the VisualizationData objects and store them
        for folder in self.folders:
            v = VisualizationData(folder)

            # Check existence of objects in dictionary before storing
            if v.comms_period not in self.viz_data_obj_dict:
                self.viz_data_obj_dict[v.comms_period] = {}

            if v.comms_prob not in self.viz_data_obj_dict[v.comms_period]:
                self.viz_data_obj_dict[v.comms_period][v.comms_prob] = {}

            if v.num_agents in self.viz_data_obj_dict[v.comms_period][v.comms_prob]:
                raise ValueError("The data for period={0}, prob={1}, num_agents={2} exists already!".format(v.comms_period, v.comms_prob, v.num_agents))
            else:
                self.viz_data_obj_dict[v.comms_period][v.comms_prob][v.num_agents] = v
                self.stored_obj_counter += 1

    def get_viz_data_obj(self, comms_period: int, comms_prob: float, num_agents: int) -> VisualizationData:
        """Get the VisualizationData object.

        Args:
            comms_period: The communication period.
            comms_prob: The communication probability.
            num_agents: The number of agents.

        Returns:
            A VisualizationData object for the specified inputs.
        """
        return self.viz_data_obj_dict[comms_period][comms_prob][num_agents]

    def save(self, filepath=None, curr_time=None):
        """Serialize the class into a pickle.
        """

        # Get current time
        if curr_time is None:
            curr_time = datetime.now().strftime("%m%d%y_%H%M%S")

        if filepath:
            root, ext = os.path.splitext(filepath)
            save_path = root + "_" + curr_time + ext
        else:
            save_path = "viz_data_group" + curr_time + ".pkl"

        with open(save_path, "wb") as fopen:
            pickle.dump(self, fopen, pickle.HIGHEST_PROTOCOL)

        print( "\nSaved VisualizationDataGroup object containing {0} items at: {1}.\n".format( self.stored_obj_counter, os.path.abspath(save_path) ) )

    @classmethod
    def load(cls, filepath):
        """Load pickled data.
        """

        with open(filepath, "rb") as fopen:
            obj = pickle.load(fopen)

        # Verify the unpickled object
        assert isinstance(obj, cls)

        return obj

def plot_heatmap_vdg(
    data_obj: VisualizationDataGroup,
    row_keys: list,
    col_keys: list,
    outer_grid_row_labels: list,
    outer_grid_col_labels: list,
    threshold: float,
    **kwargs
):
    """Plot heatmap based on a VisualizationDataGroup object. TODO: currently only considers convergence data, should provide options
    """

    # Create 2 subfigures, one for the actual grid of heatmaps while the other for the colorbar
    fig_size = (16, 12)
    fig = plt.figure(tight_layout=True, figsize=fig_size, dpi=175)
    fig.suptitle("Convergence rate for a {0} network topology (threshold: {1})".format(kwargs["comms_network_str"], threshold))

    # Create two groups: left for all the heatmaps, right for the color bar
    top_gs = fig.add_gridspec(1, 2, width_ratios=[10, 1.5])

    left_gs_group = top_gs[0].subgridspec(nrows=len(outer_grid_row_labels), ncols=len(outer_grid_col_labels), wspace=0.001)
    right_gs_group = top_gs[1].subgridspec(1, 1)

    ax_lst = left_gs_group.subplots(sharex=True, sharey=True)

    # Find heatmap minimum and maximum to create a standard range for color bar later
    heatmap_data_grid = []

    minimum = np.inf
    maximum = -np.inf

    for ind_r, row in enumerate(row_keys):
        heatmap_data_grid_row = []

        for ind_c, col in enumerate(col_keys):

            v = data_obj.get_viz_data_obj(comms_period=row, comms_prob=1.0, num_agents=col)

            # Compute the convergence timestamps and save the matrix of convergence values
            convergence_vals_matrix = generate_convergence_heatmap_data(v, threshold)
            heatmap_data_grid_row.append(convergence_vals_matrix)

            minimum = np.amin([minimum, np.amin(convergence_vals_matrix)])
            maximum = np.amax([maximum, np.amax(convergence_vals_matrix)])

        heatmap_data_grid.append(heatmap_data_grid_row)
    
    print("Heatmap minimum: {0}, maximum: {1}".format(minimum, maximum))

    # Extract the inner grid ranges (simply taken from the last VisualizationData object)
    dfr_range = v.dfr_range
    sp_range = v.sp_range

    # Plot heatmap data
    for ind_r, row in enumerate(row_keys):
        for ind_c, col in enumerate(col_keys):

            tup = heatmap(
                heatmap_data_grid[ind_r][ind_c],
                ax=ax_lst[ind_r][ind_c],
                row_label=outer_grid_row_labels[ind_r],
                col_label=outer_grid_col_labels[ind_c],
                xticks=sp_range,
                yticks=dfr_range,
                vmin=minimum,
                vmax=maximum,
                activate_outer_grid_xlabel=True if ind_r == len(outer_grid_row_labels) - 1 else False,
                activate_outer_grid_ylabel=True if ind_c == 0 else False
            )

    # Add inner grid labels
    ax_lst[-1][-1].text(20.0, 19.5, "Sensor probability\nP(b|b) = P(w|w)") # sensor probability as x label
    ax_lst[0][0].text(-5, -2, "Black tile fill ratio") # fill ratio as y label

    # Add color bar
    cbar_ax = right_gs_group.subplots(subplot_kw={"aspect": 15.0}) # add subplot with aspect ratio of 15
    cbar = plt.colorbar(tup[2], cax=cbar_ax)
    cbar.ax.set_ylabel("Convergence timestep (# of observations)") # TODO: need to have a general version

    # Save the heatmap
    fig.set_size_inches(*fig_size)
    fig.savefig("/home/khaiyichin/heatmap.png", bbox_inches="tight", dpi=300)

def generate_convergence_heatmap_data(v: VisualizationData, threshold: float):
    """Generate the heatmap data using convergence values of the informed estimates.
    """

    return np.asarray(
        [ [ v.detect_convergence(dfr, sp, threshold)[2]*v.comms_period for sp in v.sp_range ] for dfr in v.dfr_range ]
    )

def plot_heatmap_vd(data_obj: VisualizationData, row_label="infer", col_label="infer", xticks="infer", yticks="infer", ax=None, **kwargs):
    """Plot heatmap based on a VisualizationData object. TODO: BROKEN, NEEDS FIXING
    """

    # Define the labels and ticks (TODO: need to create a more generalized form)
    if row_label == "infer": row_label = "Black tile fill ratio"
    if col_label == "infer": col_label = "Sensor probability, P(b|b) = P(w|w)"

    if xticks == "infer": xticks = data_obj.sp_range
    if yticks == "infer": yticks = data_obj.dfr_range

    # Collect all the heatmap data into 2-D numpy array (for informed estimates' convergence only currently)
    heatmap_data = np.asarray(
        [ [v.detect_convergence(dfr, sp, threshold)[2] for sp in data_obj.sp_range] for dfr in data_obj.dfr_range ]
    )

    return heatmap(
        heatmap_data,
        row_label=row_label,
        col_label=col_label,
        xticks=xticks,
        yticks=yticks,
        ax=ax,
        valfmt="{x:3d}",
        **kwargs
    )

def heatmap(heatmap_data, row_label="", col_label="", xticks=[], yticks=[], ax=None, cbar_kw={}, cbarlabel="", valfmt="{x:.2f}", **kwargs):
    """Create a heatmap.

    This function plots the heatmap, but doesn't display it. Use plt.show() to display the heatmap.
    Adapted from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html.

    Args:
        row_labels: A tuple containing a string for the axis title and a list of labels for the rows (abscissa).
        col_labels: A tuple containing a string for the axis title and a list of labels for the columns (ordinate).
        heatmap_data: A 2-D numpy ndarray of values.
        cbar_kw: A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel: The label for the colorbar.  Optional.
        **kwargs: All other arguments.

    Returns:
        Heatmap figure and axis objects.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    # Initialize values for keyword arguments if non-existent
    if ("activate_outer_grid_xlabel" not in kwargs) or ("activate_outer_grid_ylabel" not in kwargs): # draw labels anyway
        kwargs["activate_outer_grid_xlabel"] = True
        kwargs["activate_outer_grid_ylabel"] = True

    # Plot the heatmap
    im = ax.imshow(heatmap_data, norm=LogNorm(vmin=kwargs["vmin"], vmax=kwargs["vmax"]), cmap="jet")

    # Show all ticks and label them with the respective list entries
    if kwargs["activate_outer_grid_xlabel"]:
        ax.set_xlabel(col_label)
        ax.xaxis.labelpad = 25

    if kwargs["activate_outer_grid_ylabel"]:
        ax.set_ylabel(row_label)
        ax.yaxis.labelpad = 25

    ax.set_xticks(np.arange(heatmap_data.shape[1]), labels=xticks)
    ax.set_yticks(np.arange(heatmap_data.shape[0]), labels=yticks)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
             rotation_mode="anchor")

    # Reduce the number of visible tick labels
    len_xticks = len( ax.get_xticks() )
    if len_xticks%2 != 0: new_len_xticks = len_xticks//2 + 1
    else: new_len_xticks = len_xticks//2

    len_yticks = len( ax.get_yticks() )
    if len_yticks%2 != 0: new_len_yticks = len_yticks//2 + 1
    else: new_len_yticks = len_yticks//2

    ax.xaxis.set_major_locator( plt.MaxNLocator( new_len_xticks ) )
    ax.yaxis.set_major_locator( plt.MaxNLocator( new_len_yticks ) )

    ax.set_xticks(np.arange(heatmap_data.shape[1]), minor=True)
    ax.set_yticks(np.arange(heatmap_data.shape[0]), minor=True)

    # Create grid lines
    ax.grid(True, which="both")
    ax.grid(which="both", color="#999999", linestyle=":")

    """
    The code below is unused for now, but useful in the future
    """

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    # texts = []
    # textcolors = ("white", "black")

    # Get the formatter in case a string is supplied
    # if isinstance(valfmt, str):
    #     valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Normalize the threshold to the images color range.
    # threshold = im.norm(heatmap_data.max())/2

    # for i in range(heatmap_data.shape[0]):
    #     for j in range(heatmap_data.shape[1]):
    #         kwargs.update(color=textcolors[int(im.norm(heatmap_data[i, j]) > threshold)])
    #         text = im.axes.text(j, i, valfmt(heatmap_data[i, j], None), **kwargs)
    #         texts.append(text)

    return fig, ax, im

def plot_timeseries(target_fill_ratio, sensor_prob, data_obj: VisualizationData, agg_data=False, convergence_thresh=CONV_THRESH):
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
    fig_x_hat, ax_x_hat = plt.subplots(2, sharex=True)
    fig_x_hat.set_size_inches(8,6)

    fig_x_bar, ax_x_bar = plt.subplots(2, sharex=True)
    fig_x_bar.set_size_inches(8,6)

    fig_x, ax_x = plt.subplots(2, sharex=True)
    fig_x.set_size_inches(8,6)

    abscissa_values_x_hat = list(range(data_obj.num_obs + 1))
    abscissa_values_x_bar = list(range(0, data_obj.num_obs + 1*data_obj.comms_period, data_obj.comms_period))
    abscissa_values_x = list(range(0, data_obj.num_obs + 1*data_obj.comms_period, data_obj.comms_period))

    # Plot for all experiments
    if not agg_data:
        for n in range(data_obj.num_exp):
            stats_obj = data_obj.stats_obj_dict[target_fill_ratio][sensor_prob]

            # Plot time evolution of local estimates and confidences
            x_hat_bounds = compute_std_bounds(stats_obj.x_hat_sample_mean[n], stats_obj.x_hat_sample_std[n])
            alpha_bounds = compute_std_bounds(stats_obj.alpha_sample_mean[n], stats_obj.alpha_sample_std[n])

            ax_x_hat[0].plot(abscissa_values_x_hat, stats_obj.x_hat_sample_mean[n], label="Exp {}".format(n))
            ax_x_hat[0].fill_between(abscissa_values_x_hat, x_hat_bounds[0], x_hat_bounds[1], alpha=0.2)

            ax_x_hat[1].plot(abscissa_values_x_hat, stats_obj.alpha_sample_mean[n], label="Exp {}".format(n))
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

        # Aggregate statistics
        agg_stats_obj = data_obj.agg_stats_dict[target_fill_ratio][sensor_prob]

        # Compute the convergence timestamps
        conv_ind_x_hat, conv_ind_x_bar, conv_ind_x = \
            data_obj.detect_convergence(target_fill_ratio, sensor_prob, convergence_thresh)

        # Plot time evolution of local estimates and confidences
        x_hat_bounds = compute_std_bounds(agg_stats_obj.x_hat_mean, agg_stats_obj.x_hat_std)
        alpha_bounds = compute_std_bounds(agg_stats_obj.alpha_mean, agg_stats_obj.alpha_std)

        ax_x_hat[0].plot(abscissa_values_x_hat, agg_stats_obj.x_hat_mean)
        ax_x_hat[0].axvline(abscissa_values_x_hat[conv_ind_x_hat], color="black", linestyle=":")
        ax_x_hat[0].axhline(target_fill_ratio, color="black", linestyle="--")
        ax_x_hat[0].fill_between(abscissa_values_x_hat, x_hat_bounds[0], x_hat_bounds[1], alpha=0.2)

        ax_x_hat[1].plot(agg_stats_obj.alpha_mean)
        ax_x_hat[1].axvline(abscissa_values_x_hat[conv_ind_x_hat], color="black", linestyle=":")
        ax_x_hat[1].fill_between(abscissa_values_x_hat, alpha_bounds[0], alpha_bounds[1], alpha=0.2)

        # Plot time evolution of social estimates and confidences
        x_bar_bounds = compute_std_bounds(agg_stats_obj.x_bar_mean, agg_stats_obj.x_bar_std)
        rho_bounds = compute_std_bounds(agg_stats_obj.rho_mean, agg_stats_obj.rho_std)

        ax_x_bar[0].plot(abscissa_values_x_bar, agg_stats_obj.x_bar_mean)
        ax_x_bar[0].axvline(abscissa_values_x_bar[conv_ind_x_bar], color="black", linestyle=":")
        ax_x_bar[0].axhline(target_fill_ratio, color="black", linestyle="--")
        ax_x_bar[0].fill_between(abscissa_values_x_bar, x_bar_bounds[0], x_bar_bounds[1], alpha=0.2)

        ax_x_bar[1].plot(abscissa_values_x_bar, agg_stats_obj.rho_mean)
        ax_x_bar[1].axvline(abscissa_values_x_bar[conv_ind_x_bar], color="black", linestyle=":")
        ax_x_bar[1].fill_between(abscissa_values_x_bar, rho_bounds[0], rho_bounds[1], alpha=0.2)

        # Plot time evolution of informed estimates and confidences
        x_bounds = compute_std_bounds(agg_stats_obj.x_mean, agg_stats_obj.x_std)
        gamma_bounds = compute_std_bounds(agg_stats_obj.gamma_mean, agg_stats_obj.gamma_std)

        ax_x[0].plot(abscissa_values_x, agg_stats_obj.x_mean)
        ax_x[0].axvline(abscissa_values_x[conv_ind_x], color="black", linestyle=":")
        ax_x[0].axhline(target_fill_ratio, color="black", linestyle="--")
        ax_x[0].fill_between(abscissa_values_x, x_bounds[0], x_bounds[1], alpha=0.2)

        ax_x[1].plot(abscissa_values_x, agg_stats_obj.gamma_mean)
        ax_x[1].axvline(abscissa_values_x[conv_ind_x], color="black", linestyle=":")
        ax_x[1].fill_between(abscissa_values_x, gamma_bounds[0], gamma_bounds[1], alpha=0.2)

    # Set axis properties
    ax_x_hat[0].set_title("Average of {0} agents' local values with 1\u03c3 bounds (fill ratio: {1}, sensor: {2})".format(data_obj.num_agents, target_fill_ratio, sensor_prob))
    ax_x_hat[0].set_ylabel("Local estimates")
    ax_x_hat[0].set_ylim(0, 1.0)
    ax_x_hat[1].set_ylim(10e-3, 10e5)
    ax_x_hat[1].set_ylabel("Local confidences")
    ax_x_hat[1].set_xlabel("Observations")
    ax_x_hat[1].set_yscale("log")

    ax_x_bar[0].set_title("Average of {0} agents' social values with 1\u03c3 bounds (fill ratio: {1}, sensor: {2})".format(data_obj.num_agents, target_fill_ratio, sensor_prob))
    ax_x_bar[0].set_ylabel("Social estimates")
    ax_x_bar[0].set_ylim(0, 1.0)
    ax_x_bar[1].set_ylim(10e-3, 10e5)
    ax_x_bar[1].set_ylabel("Social confidences")
    ax_x_bar[1].set_xlabel("Observations")
    ax_x_bar[1].set_yscale("log")

    ax_x[0].set_title("Average of {0} agents' informed values with 1\u03c3 bounds (fill ratio: {1}, sensor: {2})".format(data_obj.num_agents, target_fill_ratio, sensor_prob))
    ax_x[0].set_ylabel("Informed estimates")
    ax_x[0].set_ylim(0, 1.0)
    ax_x[1].set_ylim(10e-3, 10e5)
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
    subplot_fig.legend(handles, labels, loc="center right", bbox_to_anchor=(box_2.width*1.25, 0.5))