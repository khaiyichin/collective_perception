import os
import json
import argparse
import numpy as np
import timeit
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from abc import ABC, abstractmethod
import warnings

# Local modules
from .viz_modules import line, \
                         YMIN_DECISION, \
                         YMAX_DECISION, \
                         activate_subplot_grid_lines


def plot_timeseries(time_arr, data, args):
    """Plot the time-series data.

    Args:
        time_arr: 1-D numpy array, the common time steps used by the data.
        data: n-dimensionaly numpy array, leading to a plot with n lines.
    """

    tfr = args["tfr"]
    benchmark_param_range = args["benchmark_param_range"]
    # speed = args["speed"]
    # density = args["density"]
    num_options = args["num_options"]
    labels = args["leg_labels"]

    fig, ax = plt.subplots()

    for ind, d in enumerate(data):
        # plot single line
        line(
            line_data=[time_arr, d],
            ax=ax,
            ls="-",     # line style
            lw=1,       # line width
            label=args["leg_labels"][ind] if "leg_labels" in args else "",   # legend labels
            # marker="d", # marker type
            # ms="15",     # marker size
            # mfc=c[ind], # marker face color
            # c=c[ind]    # color
        )

    # Activate grid lines
    activate_subplot_grid_lines([ax])

    # Create legend
    ax.legend(title=args["leg_title"] if "leg_title" in args else "")

    # Create title
    ax.set_title(args["ax_title"] if "ax_title" in args else "")

    # Create axes labels
    ax.set_xlabel(args["ax_labels"][0] if "ax_labels" in args else "")
    ax.set_ylabel(args["ax_labels"][1] if "ax_labels" in args else "")

def plot_decision(decision_data, args):
    """
    Args:
        decision_data: Dict (benchmark parameter value: decision fractions list)
    """

    benchmark_str = args["benchmark_str"]
    tfr = args["tfr"]
    benchmark_param_range = args["benchmark_param_range"]
    num_trials = args["num_trials"]
    sim_steps = args["sim_steps"]
    speed = args["speed"]
    density = args["density"]
    num_options = args["num_options"]
    cbar_label = args["colorbar_label"]
    filename_param_1 = "spd{0}_den{1}_bins{2}".format(int(speed), int(density), num_options)

    # Create the figures and axes
    fig_size = (6, 4)
    fig, ax_lst = plt.subplots(
        1,
        2,
        tight_layout=True,
        figsize=fig_size,
        dpi=175,
        gridspec_kw={"width_ratios": [6*num_options/len(benchmark_param_range), 1]}
    )

    # Convert sensor probability values into IDs used for deciding marker colors (the colors are fixed for sensor probability values)
    id_lst = []

    for bp in benchmark_param_range:
        id_lst.append(
            np.digitize(bp, benchmark_param_range) # may not be the correct variable to use
        )

    # Define array of colors according to the nipy_spectral colormap
    c = plt.cm.nipy_spectral(np.array(id_lst) / (num_options - 1))

    # Check the decision fraction of each sensor probability to find the ones that clutter close to 1 (using 0.9 as a threshold)
    # cluttered_keys = set()
    # for sp_dict in decision_data.values():
    #     cluttered_keys.update((k for k, v in sp_dict.items() if v > 0.9))

    # # Add offsets to reduce clutter of points so that the markers/lines that overlap is still visible
    max_comms_rounds = max(sim_steps)
    # offset = (
    #     np.linspace(
    #         0,
    #         1,
    #         len(decision_data.keys())) * max_comms_rounds / 10 - max_comms_rounds / 20
    # ).tolist()
    # offset = [offset.pop(0) if bp in decision_data.keys() else 0.0 for _, bp in enumerate(benchmark_param_range)]
    offset = [0] # @todo: unsure if staggering is needed

    # Plot the lines for each sensor probability
    for ind, s in enumerate(benchmark_param_range):

        points = [decision_data[s][step] for step in sim_steps]

        line(
            # line_data=[np.array(sim_steps)/data_obj.comms_period + offset[ind], points],
            line_data=[np.array(sim_steps) + offset[ind], points],
            ax=ax_lst[0],
            ls="-",     # line style
            lw=1,       # line width
            marker="d", # marker type
            ms="15",     # marker size
            mfc=c[ind], # marker face color
            c=c[ind]    # color
        )

        # Add markers to those that have 100% correct decisions
        consensus_ind = [i for i, p in enumerate(points) if p == 1.0]
        if consensus_ind:
            line(
                line_data=[np.array(sim_steps[consensus_ind]), [1.0]*len(consensus_ind)],
                ax=ax_lst[0],
                marker="|", # marker type
                ms="35",    # marker size
                c=c[ind]    # color
            )

    # Add ticker formatter (mostly for the log scale that is unused for now, but may be helpful in the future)
    ticker_formatter = FuncFormatter(lambda y, _: "{:.4g}".format(y))

    ax_lst[0].set_xlabel("Communication Rounds", fontsize=14)
    ax_lst[0].set_ylabel("Fraction of Correct Decisions", fontsize=14)
    ax_lst[0].set_xticks(np.array(sim_steps))
    # ax_lst[0].set_yscale("log")
    ax_lst[0].set_ylim(bottom=0, top=YMAX_DECISION)
    ax_lst[0].xaxis.set_tick_params(which="both", labelsize=10)
    ax_lst[0].yaxis.set_tick_params(which="both", labelsize=10)
    ax_lst[0].yaxis.grid(which="both", linestyle=":")
    ax_lst[0].xaxis.grid(which="major", linestyle=":")
    ax_lst[0].yaxis.set_major_formatter(ticker_formatter)
    ax_lst[0].yaxis.set_minor_formatter(ticker_formatter)

    # Create color bar
    color_bar_img = np.array(id_lst, ndmin=2).T
    ax_lst[1].imshow(color_bar_img, aspect=2, cmap="nipy_spectral", vmin=0, vmax=num_options - 1)

    # Modify tick labels
    ax_lst[1].yaxis.set_label_position("right")
    ax_lst[1].yaxis.tick_right()
    ax_lst[1].set_xticks([])
    ax_lst[1].set_yticks(range(len(id_lst)), benchmark_param_range, fontsize=8)
    ax_lst[1].set_ylabel(cbar_label, fontsize=14)

    filename_param_2 = "tfr{0}".format(int(tfr * 1e3))

    # Save the decision plot
    filename_param = "{0}_{1}".format(filename_param_2, filename_param_1)

    fig.savefig(
        "decision_" + benchmark_str + "_" +
        "s{0}_t{1}_{2}".format(int(max(sim_steps)),
                               int(num_trials),
                               filename_param) + ".png",
        bbox_inches="tight",
        dpi=300
    )


class BenchmarkVisualizerBase(ABC):
    """
        The following variables need to be implemented in the derived class.
    """
    BENCHMARK_STR = ""
    BENCHMARK_PARAM_ABBR = ""
    benchmark_param_range = 0.0
    decision_fraction = {}
    correct_decision = 0
    tfr = 0.0

    def __init__(self):
        # parser = argparse.ArgumentParser(description="Visualize {0} data".format(benchmark_str))

        # parser.add_argument(
        #     "FOLDER",
        #     type=str,
        #     help="path to folder containing the JSON data files"
        # )
        # parser.add_argument(
        #     "TFR",
        #     required=True,
        #     type=float,
        #     help="single target fill ratio to use in plotting collective decision data"
        # )
        # # parser.add_argument( # might want to provide this from the derived class
        # #     "-param",
        # #     nargs="+",
        # #     type=float,
        # #     help="(optional) benchmark-specific parameter to use in plotting collective decision data"
        # # )
        # parser.add_argument(
        #     "--step_inc",
        #     type=int,
        #     default=1000,
        #     help="(optional) the increment in simulation steps to evaluate decisions (default: 1000)"
        # )

        # return parser
        pass

    @abstractmethod
    def load_data(self, data_folder):
        raise NotImplementedError("load_data method not implemented.")

    @abstractmethod
    def compute_correct_decision(self):
        """Compute the correct decision that the robot swarm should be making.

        This should compute an index of the correct 'bin' and assign that value to self.correct_bin.
        """
        raise NotImplementedError("compute_correct_decision method not implemented.")

    @abstractmethod
    def generate_plot(self, args=None):
        raise NotImplementedError("generate_plot method not implemented.")

    @abstractmethod
    def get_decision_data(
        self,
        target_fill_ratio: float,
        sim_steps: list,
    ):
        raise NotImplementedError("get_decision_data method not implemented.")

    def get_tfr(self):
        return self.tfr

    def get_benchmark_param_range(self):
        """Get the parameter range of the benchmark specific parameter
        """
        return self.benchmark_param_range


class Crosscombe2017Visualizer(BenchmarkVisualizerBase):

    BENCHMARK_STR = "crosscombe_2017"
    BENCHMARK_PARAM_ABBR = "frr" # flawed robot ratio

    def __init__(self, args):

        # Initialize parameters to visualize
        self.tfr = float(args.TFR)
        self.benchmark_param_range = \
            [float(i) for i in args.FRR] if isinstance(args.FRR, list) else [args.FRR]
        self.frr = self.benchmark_param_range
        self.num_flawed_agents = \
            {key: 0 for key in args.FRR} if isinstance(args.FRR, list) else {args.FRR: 0}

        # if args.U:

        self.initial_pass = {key: True for key in self.frr}

        self.load_data(args.FOLDER)

    def load_data(self, data_folder: str):
        """Load the JSON data from a given folder.

        Args:
            data_folder: Path to the folder containing all the JSON data files.
        """
        self.data = {key: None for key in self.frr}

        # Receive folder path
        for root, _, files in os.walk(data_folder):
            for f in files:

                if os.path.splitext(f)[1] == ".json":

                    # Load the JSON file
                    with open(os.path.join(root, f), "r") as file:
                        json_dict = json.load(file)

                    # Store only the data that matches the desired tfr and benchmark param
                    if (json_dict["tfr"] != self.tfr or json_dict[self.BENCHMARK_PARAM_ABBR] not in self.frr):
                        continue

                    # Initialize data
                    if self.initial_pass[json_dict["frr"]]:

                        # Store common parameters only for the first pass
                        if all(val_bool for val_bool in self.initial_pass.values()):
                            self.num_trials = json_dict["num_trials"]
                            self.num_agents = json_dict["num_agents"]
                            self.num_steps = json_dict["num_steps"]
                            self.comms_range = round(json_dict["comms_range"], 3)
                            self.speed = round(json_dict["speed"], 3)
                            self.density = round(json_dict["density"], 3)
                            self.option_qualities = json_dict["option_qualities"]
                            self.num_options = len(self.option_qualities)

                        self.num_flawed_agents[json_dict["frr"]] = json_dict["num_flawed_robots"]
                        self.data[json_dict["frr"]] = np.empty(
                            (
                                self.num_trials,
                                self.num_agents - json_dict["num_flawed_robots"],
                                self.num_steps + 1,
                                self.num_options
                            )
                        )

                        self.initial_pass[json_dict["frr"]] = False

                    # Decode json file into data
                    self.data[json_dict["frr"]][json_dict["trial_ind"]] = \
                        self.decode_beliefs(json_dict["beliefs"])

    def generate_plot(self, args=None):

        # Generate decision plot
        if args.viz_type == "decision":
            plotted_sim_steps = [i for i in range(0, self.num_steps + 1, args.step_inc)]
            decision_data = self.get_decision_data(plotted_sim_steps)

            args_plot_decision = {
                "benchmark_str": self.BENCHMARK_STR,
                "tfr": self.tfr,
                "benchmark_param_range": self.benchmark_param_range, # this is so that the plot functions can use a generic value
                "num_trials": self.num_trials,
                "sim_steps": plotted_sim_steps,
                "speed": self.speed,
                "density": self.density,
                "num_options": self.num_options,
                "colorbar_label": "Flawed Robot Ratios"
            }

            plot_decision(decision_data, args_plot_decision)

        # Generate time series plot
        if args.viz_type == "series":
            time_series_data = self.get_time_series_data()

            args_plot_series = {
                "tfr": self.tfr,
                "benchmark_param_range": self.benchmark_param_range, # this is so that the plot functions can use a generic value
                "speed": self.speed,
                "density": self.density,
                "num_options": self.num_options,
                "leg_labels": range(1, len(time_series_data)+1),
                "leg_title": "Options",
                "ax_title": "Average belief of different options with $f$={0}, $\lambda$={1}".format(self.tfr, self.frr[0]),
                "ax_labels": ["Time steps", "Average belief"]
            }

            plot_timeseries(
                np.asarray([i for i in range(self.num_steps + 1)]),
                time_series_data,
                args_plot_series
            )

        # Plot the average belief in best option w.r.t. flawed robot ratio
        if args.viz_type == "flawed_rate":

            avg_beliefs = self.get_avg_belief_vs_flawed_ratio_data()

            pass

    def get_avg_belief_vs_flawed_ratio_data(self, sim_step: int):
        """Get the average belief data with respect to the flawed robot ratio.

        Args:
            sim_step: The simulation step to collect the beliefs from.

        Returns:
            A 1-D numpy array consisting of the average beliefs at different flawed robot ratios.
        """

        for frr in self.frr:
            data = self.data[frr]
            # need to extract self.data at the `sim_step`th step
            # recall that self.data is a 4-dimensional numpy ndarray: num_trials x num_robots x num_steps x num_options
            pass

    def get_time_series_data(self):
        """Get time series data.

        Before normalization, the indeterminate beliefs have a value of 1, while the the positive
        beliefs have a value of 2.

        Returns:
            Numpy array in the form [belief_lst_for_option_1, belief_lst_for_option_2, ...]
        """

        data = self.data[self.frr[0]] # for time series data we only do 1 FRR plot at a time

        # Normalize the belief
        norm_const = np.sum(data, axis=3).reshape((*data.shape[:3], 1))

        normalized_belief_data = np.divide(data, norm_const)

        # Get the average belief for each option
        avg_belief_data = normalized_belief_data.mean(axis=1).mean(
            axis=0
        ) # shape = (num_steps x num_options)

        return np.moveaxis(avg_belief_data, 0, 1)

    def get_decision_data(self, sim_steps: list):
        """Get the decision data of all the robots.

        Args:
            sim_steps: Time step instances to get decision data from.

        Returns:
            A dict in the form {frr_1: {ss_11: decision_11, ss_12: decision_12, ...},
                                frr_2: {ss_21: decision_21, ss_22: decision_22, ...},
                                frr_3: {ss_31: decision_31, ss_32: decision_32, ...},
                                ...}
        """

        # Convert the beliefs into decisions
        if not self.decision_fraction:
            self.compute_correct_decision()
            self.convert_belief_to_decision()

        return {
            bp: {ss: self.decision_fraction[bp][ss] for ss in sim_steps
                } for bp in self.benchmark_param_range
        }

    def compute_correct_decision(self):
        self.correct_decision = np.argmax(self.option_qualities)

    def decode_beliefs(self, belief_str_vec):
        """Decodes the array of belief string.

        Args:
            belief_str_vec: List of lists of beliefs for all robots (across all time steps) in a single trial

        Returns:
            Numpy array of belief indices for the robots across all time steps
        """

        # belief_str_vec is a num_agents x num_steps list of lists
        return np.asarray([[list(map(int, elem)) for elem in row] for row in belief_str_vec])

    def convert_belief_to_decision(self):
        """Convert the belief state of the robots into a bin selection (index of the max belief).
        
        """

        for frr in self.frr:
            data = self.data[
                frr
            ] # data is a 4-dimensional numpy ndarray: num_trials x num_robots x num_steps x num_options

            # Find indices of the robot's maximum belief
            max_indices = np.argmax(data, axis=3)

            # Replace the index with -1 if the belief is indeterminate (i.e., == 1)
            decisions = np.where(
                data.max(axis=3) == 1,
                -1,
                max_indices
            ) # decisions would be 3-dimensional: num_trials x num_robots x num_steps

            # Compute the fraction of correct decisions
            total_num_nonflawed_agents = (
                self.num_agents - self.num_flawed_agents[frr]
            ) * self.num_trials

            self.decision_fraction[frr] = np.asarray(
                [
                    np.count_nonzero(decisions[:,:,i] == self.correct_decision) / total_num_nonflawed_agents
                    for i in range(self.num_steps + 1)
                ]
            ) # this is 1-D with num_steps length


class Ebert2020Visualizer(BenchmarkVisualizerBase):
    BENCHMARK_STR = "ebert_2020"
    BENCHMARK_PARAM_ABBR = "sp" # sensor probability

    def __init__(self, args, gen_plot=True):
        # Initialize parameters to visualize
        self.tfr = float(args.TFR)
        self.benchmark_param_range = \
            [float(i) for i in args.SP] if isinstance(args.SP, list) else [args.SP]
        self.sp = self.benchmark_param_range
        self.prior_param = \
            {key: 0 for key in args.SP} if isinstance(args.SP, list) else {args.SP: 0}
        self.credible_threshold = \
            {key: 0 for key in args.SP} if isinstance(args.SP, list) else {args.SP: 0}
        self.positive_feedback = \
            {key: 0 for key in args.SP} if isinstance(args.SP, list) else {args.SP: 0}
        self.collectively_decided = \
            {key: 0 for key in args.SP} if isinstance(args.SP, list) else {args.SP: 0}

        self.initial_pass = {key: True for key in self.sp}

        self.load_data(args.FOLDER)

    def load_data(self, data_folder: str):
        """Load the JSON data from a given folder.

        Args:
            data_folder: Path to the folder containing all the JSON data files.
        """
        self.data = {key: None for key in self.sp}

        # Receive folder path
        for root, _, files in os.walk(data_folder):
            for f in files:

                if os.path.splitext(f)[1] == ".json":

                    # Load the JSON file
                    with open(os.path.join(root, f), "r") as file:
                        json_dict = json.load(file)

                    # Store only the data that matches the desired tfr and benchmark param
                    if (json_dict["tfr"] != self.tfr or json_dict[self.BENCHMARK_PARAM_ABBR] not in self.sp):
                        continue

                    # Initialize data
                    if self.initial_pass[json_dict["sp"]]:

                        # Store common parameters only for the first pass
                        if all(val_bool for val_bool in self.initial_pass.values()):
                            self.num_trials = json_dict["num_trials"]
                            self.num_agents = json_dict["num_agents"]
                            self.num_steps = json_dict["num_steps"]
                            self.comms_range = round(json_dict["comms_range"], 3)
                            self.speed = round(json_dict["speed"], 3)
                            self.density = round(json_dict["density"], 3)
                            self.prior_param = json_dict["prior_param"]
                            self.credible_threshold = json_dict["credible_threshold"]
                            self.positive_feedback = json_dict["positive_feedback"]
                            self.collectively_decided[json_dict["sp"]] = json_dict["collectively_decided"]

                        self.data[json_dict["sp"]] = np.empty(
                            (
                                self.num_trials,
                                self.num_agents,
                                self.num_steps + 1,
                                1 # decision of robot
                            )
                        )

                        self.initial_pass[json_dict["sp"]] = False

                    # Warn if the data contains undecided robots
                    self.collectively_decided = json_dict["collectively_decided"]
                    if not self.collectively_decided:
                        warnings.warn("Not all robots have collectively made a decision for this file: {0}".format(f))

                    # Decode json file into data
                    self.data[json_dict["sp"]][json_dict["trial_ind"]] = \
                        self.decode_data_str(json_dict["data_str"])

    def generate_plot(self, args=None):

        if args.viz_type == "decision":
            if self.num_steps < args.step_inc: raise Exception("Step increments is too large for number of timesteps available.")

            plotted_sim_steps = [i for i in range(0, self.num_steps + 1, args.step_inc)]
            decision_data = self.get_decision_data(plotted_sim_steps)

            args_plot_decision = {
                "benchmark_str": self.BENCHMARK_STR,
                "tfr": self.tfr,
                "benchmark_param_range": self.benchmark_param_range, # this is so that the plot functions can use a generic value
                "num_trials": self.num_trials,
                "sim_steps": plotted_sim_steps,
                "speed": self.speed,
                "density": self.density,
                "num_options": 2,
                "colorbar_label": "Sensor Accuracies"
            }

            print("debugg", plotted_sim_steps, "\n", decision_data)

            plot_decision(decision_data, args_plot_decision)

    def get_decision_data(self, sim_steps: list):
        """Get the decision data of all the robots.

        Args:
            sim_steps: Time step instances to get decision data from.

        Returns:
            A dict in the form {sp_1: {ss_11: decision_11, ss_12: decision_12, ...},
                                sp_2: {ss_21: decision_21, ss_22: decision_22, ...},
                                sp_3: {ss_31: decision_31, ss_32: decision_32, ...},
                                ...}
        """

        # Convert the beliefs into decisions
        if not self.decision_fraction:
            self.compute_correct_decision()
            self.compute_decision_fraction()

        return {
            bp: {ss: self.decision_fraction[bp][ss] for ss in sim_steps
                } for bp in self.benchmark_param_range
        }

    def compute_correct_decision(self):
        self.correct_decision = 0 if self.tfr < 0.5 else 1

    def compute_decision_fraction(self):
        # self.data is a dict of sp: trials: num_agents

        # Calculate the fractions by each sp value
        total_agents = self.num_agents * self.num_trials

        for sp, np_array_data in self.data.items(): # np_array_data is num_trials x num_agents x num_steps x 1

            self.decision_fraction[sp] = np.asarray(
                [
                    np.count_nonzero(np_array_data[:,:,i] == self.correct_decision) / total_agents
                    for i in range(self.num_steps + 1)
                ]
            )

            """ Commenting the following (for now?) because we can just `ask' the robots what their decisions
            are at specific time steps. If they haven't come to a decision yet, we consider that as an `incorrect'
            decision.
            """

            # # Find the time step when each agent makes their decision
            # first_decision_indices = np.argmax(np_array_data[0] != -1, axis = 1) # this returns an index of 0 if not found any 0s or 1s

            # # Get the latest timestep where everyone has made their decision
            # last_decision_ind = max(first_decision_indices)

            # if not self.collectively_decided: # that means not all have decided
            #     last_decision_ind = self.num_steps # set to the last timestep

            # # Get the decisions of everyone at that timestep
            # decisions = np_array_data[0][:,last_decision_ind]
            # print("debug??!?!",first_decision_indices, last_decision_ind, decisions)

            # first_decision_indices = np.argmax(np_array_data[1] != -1, axis = 1) # this returns an index of 0 if not found any 0s or 1s
            # last_decision_ind = max(first_decision_indices)

            # if not self.collectively_decided: # that means not all have decided
            #     last_decision_ind = self.num_steps # set to the last timestep

            # # Get the decisions of everyone at that timestep
            # decisions = np_array_data[1][:,last_decision_ind]
            # print("debug??!?!",first_decision_indices, last_decision_ind, decisions)


    def decode_data_str(self, data_str_vec):
        """Decodes the array of data string.

        Args:
            data_str_vec: List of lists of data string for all robots (across all time steps) in a single trial

        Returns:
            Numpy array of decisions for the robots across all time steps (num_agents x num_timesteps x 1)
        """
        # data_str_vec is a num_agents x num_steps list of lists
        return np.asarray([[[int(elem.split(",")[3])] for elem in row] for row in data_str_vec]) # the 4th element is the decision

class Visualizer:

    def __init__(self):
        parser = argparse.ArgumentParser(description="Visualize benchmark data")

        parser.add_argument(
            "FOLDER",
            type=str,
            help="path to folder containing the JSON data files"
        )
        parser.add_argument(
            "TFR",
            type=float,
            help="single target fill ratio to use in plotting collective decision data"
        )
        parser.add_argument(
            "--step_inc",
            type=int,
            default=1000,
            help="(optional) the increment in simulation steps to evaluate decisions (default: 1000)"
        )
        parser.add_argument(
            "-s",
            action="store_true",
            help="flag to show the plots"
        )
        benchmark_type_subparser = parser.add_subparsers(
            dest="benchmark",
            required=True,
            help="benchmark type"
        )

        # Crosscombe 2017 arguments
        crosscombe_2017_subparser = benchmark_type_subparser.add_parser(
            Crosscombe2017Visualizer.BENCHMARK_STR,
            help="{0} benchmark".format(Crosscombe2017Visualizer.BENCHMARK_STR)
        )

        crosscombe_2017_viz_type_subparser = crosscombe_2017_subparser.add_subparsers(
            dest="viz_type",
            required=True,
            help="commands for visualization type"
        )

        crosscombe_2017_viz_type_decision_subparser = crosscombe_2017_viz_type_subparser.add_parser(
            "decision",
            help="visualize collective-decision making data"
        )
        crosscombe_2017_viz_type_series_subparser = crosscombe_2017_viz_type_subparser.add_parser(
            "series",
            help="visualize time series data"
        )

        crosscombe_2017_viz_type_decision_subparser.add_argument( # might want to provide this from the derived class
            Crosscombe2017Visualizer.BENCHMARK_PARAM_ABBR.upper(),
            nargs="+",
            type=float,
            help="flawed robot ratios (space-delimited array) to use in plotting collective decision data"
        )

        crosscombe_2017_viz_type_series_subparser.add_argument( # might want to provide this from the derived class
            Crosscombe2017Visualizer.BENCHMARK_PARAM_ABBR.upper(),
            type=float,
            help="flawed robot ratio to use in plotting time series data"
        )

        # Ebert 2020 arguments
        ebert_2020_subparser = benchmark_type_subparser.add_parser(
            Ebert2020Visualizer.BENCHMARK_STR,
            help="{0} benchmark".format(Ebert2020Visualizer.BENCHMARK_STR)
        )

        ebert_2020_viz_type_subparser = ebert_2020_subparser.add_subparsers(
            dest="viz_type",
            required=True,
            help="commands for visualization type"
        )

        ebert_2020_viz_type_decision_subparser = ebert_2020_viz_type_subparser.add_parser(
            "decision",
            help="visualize collective-decision making data"
        )

        ebert_2020_viz_type_decision_subparser.add_argument( # might want to provide this from the derived class
            Ebert2020Visualizer.BENCHMARK_PARAM_ABBR.upper(),
            nargs="+",
            type=float,
            help="sensor probabilities (space-delimited array) to use in plotting collective decision data"
        )

        args = parser.parse_args()

        # Initialize visualizer objects
        if args.benchmark == Crosscombe2017Visualizer.BENCHMARK_STR:
            self.benchmark_visualizer = Crosscombe2017Visualizer(args)

        elif args.benchmark == Ebert2020Visualizer.BENCHMARK_STR:
            self.benchmark_visualizer = Ebert2020Visualizer(args)

        # Generate plots
        start = timeit.default_timer()

        self.benchmark_visualizer.generate_plot(args)

        end = timeit.default_timer()

        print('Elapsed time:', end - start)

        if args.s:
            plt.show()