import os
import json
import argparse
import numpy as np
import timeit
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from abc import ABC, abstractmethod

# Local modules
from .viz_modules import line, YMIN_DECISION, YMAX_DECISION


def plot_time_series(time_arr, data, args):
    """Plot the time-series data.

    Args:
        time_arr: 1-D numpy array, the common time steps used by the data.
        data: n-dimensionaly numpy array, leading to a plot with n lines.
    """

    tfr = args["tfr"]
    benchmark_param_range = args["benchmark_param_range"]
    bins = args["bins"]
    speed = args["speed"]
    density = args["density"]
    num_options = args["num_options"]
    # filename_param_1 = "spd{0}_den{1}_bins{2}".format(int(speed), int(density), bins)

    fig, ax = plt.subplots()

    # Convert options into IDs used for deciding marker colors
    id_lst = []

    for bp in range(num_options):
        id_lst.append(
            np.digitize(bp, range(num_options)) # may not be the correct variable to use
        )

    # Define array of colors according to the nipy_spectral colormap
    # c = plt.cm.nipy_spectral(np.array(id_lst) / (num_options - 1))

    # print("debug", data.shape)

    for ind, d in enumerate(data):
        # plot single line
        line(
            line_data=[time_arr, d],
            ax=ax,
            ls="-",     # line style
            lw=1,       # line width
            # marker="d", # marker type
            # ms="15",     # marker size
            # mfc=c[ind], # marker face color
            # c=c[ind]    # color
        )


def plot_decision(decision_data, args):
    """
    Args:
        decision_data: Dict (benchmark parameter value: decision fractions list)
    """

    tfr = args["tfr"]
    benchmark_param_range = args["benchmark_param_range"]
    bins = args["bins"]
    sim_steps = args["sim_steps"]
    speed = args["speed"]
    density = args["density"]
    num_options = args["num_options"]
    cbar_label = args["colorbar_label"]
    filename_param_1 = "spd{0}_den{1}_bins{2}".format(int(speed), int(density), bins)

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
    offset = (
        np.linspace(
            0,
            1,
            len(decision_data.keys())) * max_comms_rounds / 10 - max_comms_rounds / 20
    ).tolist()
    # offset = [offset.pop(0) if bp in decision_data.keys() else 0.0 for _, bp in enumerate(benchmark_param_range)]

    # Plot the lines for each sensor probability
    for ind, s in enumerate(benchmark_param_range):

        points = decision_data[s]

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

    # fig.savefig(
    #     "decision_" + data_obj_type + "_" +
    #     "s{0}_t{1}_{2}".format(int(data_obj.num_steps),
    #                            int(data_obj.num_trials),
    #                            filename_param) + ".png",
    #     bbox_inches="tight",
    #     dpi=300
    # )


class BenchmarkVisualizerBase(ABC):
    """
        The following variables need to be implemented in the derived class.
    """
    BENCHMARK_STR = ""
    BENCHMARK_PARAM_ABBR = ""
    benchmark_param_range = 0.0
    decision_fraction = {}
    correct_decision = 0

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

    def get_tfr_range(self):
        pass

    def get_param_range(self):
        """Get the parameter range of the benchmark specific parameter
        """
        pass


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

        start = timeit.default_timer()

        self.generate_plot(args)

        end = timeit.default_timer()

        print('Elapsed time:', end - start)

        if args.s:
            plt.show()

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
                "tfr": self.tfr,
                "benchmark_param_range": self.benchmark_param_range, # this is so that the plot functions can use a generic value
                "bins": self.num_options,
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
                "bins": self.num_options,
                "speed": self.speed,
                "density": self.density,
                "num_options": self.num_options
            }

            print("debug", time_series_data.shape, time_series_data.sum(axis=0))

            plot_time_series(
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
            A dict in the form {frr_1: decision_lst_1, frr_2: decision_lst_2, ...}
        """

        # Convert the beliefs into decisions
        if not self.decision_fraction:
            self.compute_correct_decision()
            self.convert_belief_to_decision()

        return {
            bp: [self.decision_fraction[bp][ss] for ss in sim_steps
                ] for bp in self.benchmark_param_range
        }

    def compute_correct_decision(self):
        self.correct_decision = np.argmax(self.option_qualities)

    def decode_beliefs(self, belief_str_vec):
        """Decodes the array of belief string.

        Args:
            belief_str_vec: List of lists of beliefs for all robots (across all time steps) in a single trial

        Returns:
            Numpy array of belief indices for the robot across all time steps
        """

        # belief_str_vec is a num_agents x num_steps list of lists
        return np.asarray([[list(map(int, elem)) for elem in row] for row in belief_str_vec])

    # def convert_belief_to_decision(self, sim_steps):
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
            help="flawed robot ratio (space-delimited array) to use in plotting collective decision data"
        )

        crosscombe_2017_viz_type_series_subparser.add_argument( # might want to provide this from the derived class
            Crosscombe2017Visualizer.BENCHMARK_PARAM_ABBR.upper(),
            type=float,
            help="flawed robot ratio to use in plotting time series data"
        )

        args = parser.parse_args()

        if args.benchmark == Crosscombe2017Visualizer.BENCHMARK_STR:
            self.benchmark_visualizer = Crosscombe2017Visualizer(args)
