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
from abc import ABC, abstractmethod

from sim_modules import ExperimentData, Sim
import simulation_set_pb2

# Default values
CONV_THRESH = 5e-3
FIG_SIZE = (20, 12)

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

    def __init__(self, exp_data_obj_folder=""):

        first_obj = True

        # Iterate through folder contents recursively to obtain all serialized filenames
        exp_data_obj_paths = []

        for root, _, files in os.walk(exp_data_obj_folder):
            for f in files:
                if os.path.splitext(f)[1] == ".pkl" or os.path.splitext(f)[1] == ".pbs":
                    exp_data_obj_paths.append( os.path.join(root, f) )

        # Load the files
        for path in exp_data_obj_paths:

            # Check type of file to load
            if os.path.splitext(path)[1] == ".pkl": obj = self.load_pkl_file(path)
            elif os.path.splitext(path)[1] == ".pbs": obj = self.load_proto_file(path)
            else: raise RuntimeError("Unknown extension encountered; please provide \".pkl\" or \".pbs\" files.")

            # Check that common simulation parameters are the same before combining
            if first_obj:

                # Check simulation type
                if hasattr(obj, "sim_type"): # TODO: legacy ExperimentData classes has no sim_type; remove this when upgrade is complete
                    self.sim_type = obj.sim_type

                    if self.sim_type == "dynamic":
                        self.comms_range = obj.comms_range
                        self.density = obj.density
                        self.speed = obj.speed
                        self.comms_period = 1

                    elif self.sim_type == "static":
                        self.graph_type = obj.graph_type
                        self.comms_period = obj.comms_period
                        self.comms_prob = obj.comms_prob

                else: # TODO: legacy ExperimentData classes has no sim_type; remove this when upgrade is complete
                    self.graph_type = obj.graph_type
                    self.comms_period = obj.comms_period
                    self.comms_prob = obj.comms_prob

                # Common parameters
                self.num_agents = obj.num_agents
                self.num_exp = obj.num_exp
                self.num_obs = obj.num_obs
                self.dfr_range = obj.dfr_range
                self.sp_range = obj.sp_range
                self.stats_obj_dict = obj.stats_obj_dict
                self.agg_stats_dict = {} # to be populated later
                
                first_obj = False

            else:
                similarity_bool = True

                if hasattr(self, "sim_type"): # TODO: legacy ExperimentData classes has no sim_type; remove this when upgrade is complete
                    similarity_bool = similarity_bool and (self.sim_type == obj.sim_type)

                    if self.sim_type == "dynamic":
                        similarity_bool = similarity_bool and (self.comms_range == obj.comms_range)
                        similarity_bool = similarity_bool and (self.density == obj.density)
                        similarity_bool = similarity_bool and (self.speed == obj.speed)

                    elif self.sim_type == "static":
                        similarity_bool = similarity_bool and (self.graph_type == obj.graph_type)
                        similarity_bool = similarity_bool and (self.comms_period == obj.comms_period)
                        similarity_bool = similarity_bool and (self.comms_prob == obj.comms_prob)

                else: # TODO: legacy ExperimentData classes has no sim_type; remove this when upgrade is complete
                    similarity_bool = similarity_bool and (self.graph_type == obj.graph_type)
                    similarity_bool = similarity_bool and (self.comms_period == obj.comms_period)
                    similarity_bool = similarity_bool and (self.comms_prob == obj.comms_prob)

                # Common parameters
                similarity_bool = similarity_bool and (self.num_agents == obj.num_agents)
                similarity_bool = similarity_bool and (self.num_exp == obj.num_exp)
                similarity_bool = similarity_bool and (self.num_obs == obj.num_obs)

                # Check to see if either target fill ratio range or sensor probability range matches
                if (self.sp_range == obj.sp_range): # sensor probability matches

                    # Update the dictionary of tfr-sp_dict key-value pair
                    self.dfr_range.extend(obj.dfr_range)
                    self.stats_obj_dict.update(obj.stats_obj_dict)

                elif (self.dfr_range == obj.dfr_range): # target fill ratio matches

                    # Update the internal dictionary of sp-stats key-value pair
                    self.sp_range.extend(obj.sp_range)
                    [self.stats_obj_dict[i].update(obj.stats_obj_dict[i]) for i in obj.dfr_range]

                else: similarity_bool = False

                # Ensure that all the high level parameters are the same
                if not similarity_bool:
                    raise RuntimeError("The objects in the \"{0}\" directory do not have the same parameters.".format(exp_data_obj_paths))

        self.dfr_range = sorted(self.dfr_range)
        self.sp_range = sorted(self.sp_range)

        self.aggregate_statistics()

    def load_pkl_file(self, folder_path): return ExperimentData.load(folder_path, False)

    def load_proto_file(self, folder_path):
        """Load SimulationStatsSet protobuf file into VisualizationData object.
        """
        with open(folder_path, "rb") as fopen:
            sim_stats_set_msg = simulation_set_pb2.SimulationStatsSet()
            sim_stats_set_msg.ParseFromString(fopen.read())

        # Add dynamic class member variables (remove abstraction layers)
        setattr(simulation_set_pb2.SimulationStatsSet, "sim_type", sim_stats_set_msg.sim_set.sim_type)
        setattr(simulation_set_pb2.SimulationStatsSet, "num_agents", sim_stats_set_msg.sim_set.num_agents)
        setattr(simulation_set_pb2.SimulationStatsSet, "num_exp", sim_stats_set_msg.sim_set.num_trials)
        setattr(simulation_set_pb2.SimulationStatsSet, "dfr_range", np.round(sim_stats_set_msg.sim_set.tfr_range, 3).tolist())
        setattr(simulation_set_pb2.SimulationStatsSet, "sp_range", np.round(sim_stats_set_msg.sim_set.sp_range, 3).tolist())
        setattr(simulation_set_pb2.SimulationStatsSet, "num_obs", sim_stats_set_msg.sim_set.num_steps)
        setattr(simulation_set_pb2.SimulationStatsSet, "comms_range", sim_stats_set_msg.sim_set.comms_range)
        setattr(simulation_set_pb2.SimulationStatsSet, "speed", sim_stats_set_msg.sim_set.speed)
        setattr(simulation_set_pb2.SimulationStatsSet, "density", sim_stats_set_msg.sim_set.density)
        setattr(simulation_set_pb2.SimulationStatsSet, "stats_obj_dict", {i: {j: None for j in sim_stats_set_msg.sp_range} for i in sim_stats_set_msg.dfr_range} )

        for stats_packet in sim_stats_set_msg.stats_packets:
            stats_obj = Sim.SimStats(None)

            # Extract Stats protobuf message into a list of 4-tuples containing (x_mean, conf_mean, x_std, conf_std) for all trials
            repeated_trial_local_vals = [ ( np.asarray(s.x_mean),
                                            np.asarray(s.conf_mean),
                                            np.asarray(s.x_std),
                                            np.asarray(s.conf_std) ) for s in stats_packet.rts.local_vals ]
            repeated_trial_social_vals = [ ( np.asarray(s.x_mean),
                                            np.asarray(s.conf_mean),
                                            np.asarray(s.x_std),
                                            np.asarray(s.conf_std) ) for s in stats_packet.rts.social_vals ]
            repeated_trial_informed_vals = [ ( np.asarray(s.x_mean),
                                            np.asarray(s.conf_mean),
                                            np.asarray(s.x_std),
                                            np.asarray(s.conf_std) ) for s in stats_packet.rts.informed_vals ]

            stats_obj.x_hat_sample_mean, stats_obj.alpha_sample_mean, stats_obj.x_hat_sample_std, stats_obj.alpha_sample_std = zip(*repeated_trial_local_vals)
            stats_obj.x_bar_sample_mean, stats_obj.rho_sample_mean, stats_obj.x_bar_sample_std, stats_obj.rho_sample_std = zip(*repeated_trial_social_vals)
            stats_obj.x_sample_mean, stats_obj.gamma_sample_mean, stats_obj.x_sample_std, stats_obj.gamma_sample_std = zip(*repeated_trial_informed_vals)



            #### TODO NEED FIXING: REMOVE AFTER LEGACY DATA IS FIXED (RE-REPLICATED WITH UPDATED PROGRAM) ####
            # Temporary fix to bug that computes additional value in the beginning
            if (len(stats_obj.x_hat_sample_mean)>1) and (len(stats_obj.x_hat_sample_mean[0]) != len(stats_obj.x_hat_sample_mean[1])):
                print("DEBUG: DEVELOPER TAKE NOTE TO FIX THIS; THE VERY FIRST TRIAL HAS AN ADDITIONAL SAMPLE")

                # Have to do this because of numpy doesn't allow inplace changing, and tuples do not allow reassignment
                stats_obj.x_hat_sample_mean = [ *stats_obj.x_hat_sample_mean ]
                stats_obj.alpha_sample_mean = [ *stats_obj.alpha_sample_mean ]
                stats_obj.x_hat_sample_std = [ *stats_obj.x_hat_sample_std ]
                stats_obj.alpha_sample_std = [ *stats_obj.alpha_sample_std ]

                stats_obj.x_bar_sample_mean = [ *stats_obj.x_bar_sample_mean ]
                stats_obj.rho_sample_mean = [ *stats_obj.rho_sample_mean ]
                stats_obj.x_bar_sample_std = [ *stats_obj.x_bar_sample_std ]
                stats_obj.rho_sample_std = [ *stats_obj.rho_sample_std ]

                stats_obj.x_sample_mean = [ *stats_obj.x_sample_mean ]
                stats_obj.gamma_sample_mean = [ *stats_obj.gamma_sample_mean ]
                stats_obj.x_sample_std = [ *stats_obj.x_sample_std ]
                stats_obj.gamma_sample_std = [ *stats_obj.gamma_sample_std ]

                stats_obj.x_hat_sample_mean[0] = stats_obj.x_hat_sample_mean[0][1:]
                stats_obj.alpha_sample_mean[0] = stats_obj.alpha_sample_mean[0][1:]
                stats_obj.x_hat_sample_std[0] = stats_obj.x_hat_sample_std[0][1:]
                stats_obj.alpha_sample_std[0] = stats_obj.alpha_sample_std[0][1:]

                stats_obj.x_bar_sample_mean[0] = stats_obj.x_bar_sample_mean[0][1:]
                stats_obj.rho_sample_mean[0] = stats_obj.rho_sample_mean[0][1:]
                stats_obj.x_bar_sample_std[0] = stats_obj.x_bar_sample_std[0][1:]
                stats_obj.rho_sample_std[0] = stats_obj.rho_sample_std[0][1:]

                stats_obj.x_sample_mean[0] = stats_obj.x_sample_mean[0][1:]
                stats_obj.gamma_sample_mean[0] = stats_obj.gamma_sample_mean[0][1:]
                stats_obj.x_sample_std[0] = stats_obj.x_sample_std[0][1:]
                stats_obj.gamma_sample_std[0] = stats_obj.gamma_sample_std[0][1:]

            ################################

            # Convert tuple of 1-D numpy arrays into 2-D ndarrays with row = trial index and col = step index
            stats_obj.x_hat_sample_mean = np.array( [*stats_obj.x_hat_sample_mean] )
            stats_obj.alpha_sample_mean = np.array( [*stats_obj.alpha_sample_mean] )
            stats_obj.x_hat_sample_std = np.array( [*stats_obj.x_hat_sample_std] )
            stats_obj.alpha_sample_std = np.array( [*stats_obj.alpha_sample_std] )

            stats_obj.x_bar_sample_mean = np.array( [*stats_obj.x_bar_sample_mean] )
            stats_obj.rho_sample_mean = np.array( [*stats_obj.rho_sample_mean] )
            stats_obj.x_bar_sample_std = np.array( [*stats_obj.x_bar_sample_std] )
            stats_obj.rho_sample_std = np.array( [*stats_obj.rho_sample_std] )

            stats_obj.x_sample_mean = np.array( [*stats_obj.x_sample_mean] )
            stats_obj.gamma_sample_mean = np.array( [*stats_obj.gamma_sample_mean] )
            stats_obj.x_sample_std = np.array( [*stats_obj.x_sample_std] )
            stats_obj.gamma_sample_std = np.array( [*stats_obj.gamma_sample_std] )

            sim_stats_set_msg.stats_obj_dict[np.round(stats_packet.packet.tfr, 3)][np.round(stats_packet.packet.b_prob, 3)] = stats_obj

        return sim_stats_set_msg

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

    def compute_accuracy(self, target_fill_ratio: float, sensor_prob: float, conv_ind_lst=None):
        """Compute the estimate accuracy with respect to the target fill ratio.

        If the list of convergence indices is not provided, then convergence indices will be
        found in this function to use as target values to compute accuracies, based on the
        default convergence threshold.

        Args:
            target_fill_ratio: The target fill ratio used in the simulation.
            sensor_prob: The sensor probability used in the simulation.
            conv_ind_lst: The list of 3 convergence indices.

        Returns:
            The 3 accuracies.
        """

        if conv_ind_lst is None:
            conv_ind_lst = self.detect_convergence(target_fill_ratio, sensor_prob)

        acc_x_hat = abs(self.agg_stats_dict[target_fill_ratio][sensor_prob].x_hat_mean[conv_ind_lst[0]] - target_fill_ratio)
        acc_x_bar = abs(self.agg_stats_dict[target_fill_ratio][sensor_prob].x_bar_mean[conv_ind_lst[1]] - target_fill_ratio)
        acc_x = abs(self.agg_stats_dict[target_fill_ratio][sensor_prob].x_mean[conv_ind_lst[2]] - target_fill_ratio)

        return acc_x_hat, acc_x_bar, acc_x

    def get_informed_estimate_metrics(self, target_fill_ratio: float, sensor_prob: float, threshold=CONV_THRESH):
        """Get the accuracy and convergence of informed estimates.

        Args:
            target_fill_ratio: The target fill ratio used in the simulations.
            sensor_prob: The sensor probability used in the simulation.
            threshold: A float parametrizing the difference threshold.

        Returns:
            A tuple of convergence (taking communication period into account) and accuracy values for the informed estimate.
        """

        conv_lst = self.detect_convergence(target_fill_ratio, sensor_prob, threshold)
        acc_lst = self.compute_accuracy(target_fill_ratio, sensor_prob, conv_lst)

        return conv_lst[2]*self.comms_period, acc_lst[2]

class VisualizationDataGroupBase(ABC):

    def __init__(self, data_folder, ext):
        self.viz_data_obj_dict = {}
        self.stored_obj_counter = 0

        # Iterate through folder contents recursively to obtain folders containing the serialized files
        exp_data_obj_folders = []

        for root, _, files in os.walk(data_folder):

            # Check to see if any pickle file exist in the current directory
            serialized_files = [f for f in files if os.path.splitext(f)[1] == ext]

            if len(serialized_files) == 0: # move on to the next folder
                continue
            else: # file found, that means the directory two levels up is needed; the first level only describes the inner simulation parameters (e.g., tfr or sp)
                parent_folder, folder = os.path.split( os.path.abspath(root) )
                exp_data_obj_folders.append(parent_folder)

        self.folders = list(set(exp_data_obj_folders)) # store the unique values for the paths

    @abstractmethod
    def get_viz_data_obj(self, args: dict) -> VisualizationData:
        """Get VisualizationData object.
        """
        raise NotImplementedError("get_viz_data_obj function not implemented.")

    @abstractmethod
    def save(self, filepath=None, curr_time=None):
        """Save to pickle.
        """
        raise NotImplementedError("save function not implemented.")

    @classmethod
    def load(cls, filepath):
        """Load pickled data.
        """

        with open(filepath, "rb") as fopen:
            obj = pickle.load(fopen)

        # Verify the unpickled object
        assert isinstance(obj, cls)

        return obj

class VisualizationDataGroupStatic(VisualizationDataGroupBase):
    """Class to store VisualizationData objects for static simulations.

    The VisualizationData objects are stored by first using the VisualizationData's method
    in loading serialized ExperimentData files. Then each VisualizationData object
    are stored in this class.

    This class is intended to store VisualizationData objects with the same:
        - communication network type, and
        - number of experiments,
    and with varying:
        - communication period,
        - communication probability,
        - number of agents.
    The target fill ratios and sensor probabilities are already varied (with fixed ranges) in the
    stored VisualizationData objects.

    This class is initialized with 1 argument:
        data_folder: A string specifying the directory containing all the ExperimentData files.
    """

    def __init__(self, data_folder):

        super().__init__(data_folder, ".pkl")

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

    def get_viz_data_obj(self, args: dict) -> VisualizationData:
        """Get the VisualizationData object.

        Args:
            comms_period: The communication period.
            comms_prob: The communication probability.
            num_agents: The number of agents.

        Returns:
            A VisualizationData object for the specified inputs.
        """
        comms_period = args["comms_period"]
        comms_prob = args["comms_prob"]
        num_agents = args["num_agents"]

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
            save_path = "viz_data_group_static_" + curr_time + ".pkl"

        with open(save_path, "wb") as fopen:
            pickle.dump(self, fopen, pickle.HIGHEST_PROTOCOL)

        print( "\nSaved VisualizationDataGroupStatic object containing {0} items at: {1}.\n".format( self.stored_obj_counter, os.path.abspath(save_path) ) )

# TODO: legacy class; must remove after upgrade complete
class VisualizationDataGroup(VisualizationDataGroupStatic):
    pass

class VisualizationDataGroupDynamic(VisualizationDataGroupBase):
    """Class to store VisualizationData objects for dynamic simulations.

    The VisualizationData objects are stored by first using the VisualizationData's method
    in loading serialized SimulationStatSet protobuf files. Then each VisualizationData object
    are stored in this class.

    This class is intended to store VisualizationData objects with the same:
        - number of experiments,
    and with varying:
        - robot speed, and
        - number of agents.
    The target fill ratios and sensor probabilities are already varied (with fixed ranges) in the
    stored VisualizationData objects.

    This class is initialized with 1 argument:
        data_folder: A string specifying the directory containing all the SimulationStatSet protobuf files.
    """

    def __init__(self, data_folder):

        super().__init__(data_folder, ".pbs")

        # Load the VisualizationData objects and store them
        for folder in self.folders:
            v = VisualizationData(folder)

            # Check existence of objects in dictionary before storing
            if v.speed not in self.viz_data_obj_dict:
                self.viz_data_obj_dict[v.speed] = {}

            if v.num_agents in self.viz_data_obj_dict[v.speed]:
                raise ValueError("The data for speed={0}, num_agents={1} exists already!".format(v.speed, v.num_agents))
            else:
                self.viz_data_obj_dict[v.speed][v.num_agents] = v
                self.stored_obj_counter += 1

    def get_viz_data_obj(self, args: dict) -> VisualizationData:
        """Get the VisualizationData object.

        Args:
            num_agents: The number of agents.
            speed: The robot speed.

        Returns:
            A VisualizationData object for the specified inputs.
        """
        num_agents = args["num_agents"]
        speed = args["speed"]

        return self.viz_data_obj_dict[speed][num_agents]

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
            save_path = "viz_data_group_dynamic_" + curr_time + ".pkl"

        with open(save_path, "wb") as fopen:
            pickle.dump(self, fopen, pickle.HIGHEST_PROTOCOL)

        print( "\nSaved VisualizationDataGroupDynamic object containing {0} items at: {1}.\n".format( self.stored_obj_counter, os.path.abspath(save_path) ) )

def plot_heatmap_vd(data_obj: VisualizationData, threshold: float, **kwargs):
    """Plot single heatmap based on a VisualizationData object. TODO: BROKEN, NEEDS FIXING
    """

    fig_size = FIG_SIZE
    fig = plt.figure(tight_layout=True, figsize=fig_size, dpi=175)
    if kwargs["title"]: fig.suptitle(kwargs["title"], fontsize=20)

    # Create two groups: left for all the heatmaps, right for the color bar
    top_lvl_gs = fig.add_gridspec(1, 2, width_ratios=[10, 2.5])
    left_gs_group = top_lvl_gs[0].subgridspec(nrows=1, ncols=1, wspace=0.001)
    right_gs_group = top_lvl_gs[1].subgridspec(1, 1)

    ax = left_gs_group.subplots()

    # Find heatmap minimum and maximum to create a standard range for color bar later
    conv_min = np.inf
    conv_max = -np.inf
    acc_min = np.inf
    acc_max = -np.inf

    # Compute the convergence timestamps and save the matrix of convergence values
    heatmap_data = generate_combined_heatmap_data(data_obj, threshold, order=(None, "convergence", "accuracy"))

    conv_min = np.amin([conv_min, np.amin(heatmap_data[1])])
    conv_max = np.amax([conv_max, np.amax(heatmap_data[1])])
    acc_min = np.amin([acc_min, np.amin(heatmap_data[2])])
    acc_max = np.amax([acc_max, np.amax(heatmap_data[2])])

    print("Convergence timestep minimum: {0}, maximum: {1}".format(conv_min, conv_max))
    print("Accuracy error minimum: {0}, maximum: {1}".format(acc_min, acc_max))

    # Extract the target fill ratio and sensor probability ranges (simply taken from the last VisualizationData object)
    tfr_range = data_obj.dfr_range
    sp_range = data_obj.sp_range

    tup = heatmap(
        convert_to_img(heatmap_data, [(0,0), (conv_min, conv_max), (acc_min, acc_max)], active_channels=[1, 2]), # normalize and convert to img
        ax=ax,
        row_label="On-fire tile fill ratio",
        col_label="Sensor probability\nP(b|b) = P(w|w)",
        xticks=sp_range,
        yticks=tfr_range
    )

    # Add color legend
    color_leg_ax = right_gs_group.subplots(subplot_kw={"aspect": 40.0})
    color_leg_px_count = (60, 240)
    r = np.zeros((color_leg_px_count[1], color_leg_px_count[0]))
    g = np.repeat( np.reshape( np.linspace(1.0, 0.0, color_leg_px_count[1]), (color_leg_px_count[1], 1) ), color_leg_px_count[0], axis=1 )
    b = np.repeat( np.reshape( np.linspace(0.0, 1.0, color_leg_px_count[0]), (1, color_leg_px_count[0]) ), color_leg_px_count[1], axis=0)

    # Add color legend labels
    color_leg_ax.imshow( np.moveaxis( np.array( [r, g, b] ), 0, 2) )
    color_leg_ax.set_ylabel("Convergence", fontsize=15)
    color_leg_ax.set_xlabel("Accuracy", fontsize=15)

    # Add label limits for convergence
    color_leg_ax.text(-0.1*color_leg_px_count[0], 0.995*color_leg_px_count[1], "Slow", fontsize=12, rotation=90)
    color_leg_ax.text(-0.1*color_leg_px_count[0], 0.1*color_leg_px_count[0], "Fast", fontsize=12, rotation=90)

    # Add label limits for accuracy
    color_leg_ax.text(-0.01*color_leg_px_count[0], 1.02*color_leg_px_count[1], "Low", fontsize=12)
    color_leg_ax.text(0.86*color_leg_px_count[0], 1.02*color_leg_px_count[1], "High", fontsize=12)

    # Remove color legend
    color_leg_ax.set_xticks([])
    color_leg_ax.set_yticks([])

    # Save the heatmap
    fig.set_size_inches(*fig_size)
    fig.savefig("/home/khaiyichin/heatmap_single.png", bbox_inches="tight", dpi=300)

def plot_heatmap_vdg(
    data_obj: VisualizationDataGroupBase,
    row_arg_str: str,
    row_keys: list,
    col_arg_str: str,
    col_keys: list,
    outer_grid_row_labels: list,
    outer_grid_col_labels: list,
    threshold: float,
    **kwargs
):
    """Plot gridded heatmap based on a VisualizationDataGroup object.
    """

    # Create 2 subfigures, one for the actual grid of heatmaps while the other for the colorbar
    fig_size = FIG_SIZE
    fig = plt.figure(tight_layout=True, figsize=fig_size, dpi=175)
    if kwargs["title"]: fig.suptitle(kwargs["title"], fontsize=20)

    # Create two groups: left for all the heatmaps, right for the color bar
    top_gs = fig.add_gridspec(1, 2, width_ratios=[10, 2.5])

    left_gs_group = top_gs[0].subgridspec(nrows=len(outer_grid_row_labels), ncols=len(outer_grid_col_labels), wspace=0.001)
    right_gs_group = top_gs[1].subgridspec(1, 1)

    ax_lst = left_gs_group.subplots(sharex=True, sharey=True)

    # Find heatmap minimum and maximum to create a standard range for color bar later
    heatmap_data_grid = []

    conv_min = np.inf
    conv_max = -np.inf
    acc_min = np.inf
    acc_max = -np.inf

    for ind_r, row in enumerate(row_keys):
        heatmap_data_grid_row = []

        for ind_c, col in enumerate(col_keys):

            v = data_obj.get_viz_data_obj({row_arg_str: row, col_arg_str: col, "comms_prob": 1.0,}) # keep communication probabily args

            # Compute the convergence timestamps and save the matrix of convergence values
            matrix = generate_combined_heatmap_data(v, threshold, order=(None, "convergence", "accuracy"))
            heatmap_data_grid_row.append(matrix)

            conv_min = np.amin([conv_min, np.amin(matrix[1])])
            conv_max = np.amax([conv_max, np.amax(matrix[1])])
            acc_min = np.amin([acc_min, np.amin(matrix[2])])
            acc_max = np.amax([acc_max, np.amax(matrix[2])])

        heatmap_data_grid.append(heatmap_data_grid_row)
    
    print("Convergence timestep minimum: {0}, maximum: {1}".format(conv_min, conv_max))
    print("Accuracy error minimum: {0}, maximum: {1}".format(acc_min, acc_max))

    # Extract the inner grid ranges (simply taken from the last VisualizationData object)
    dfr_range = v.dfr_range
    sp_range = v.sp_range

    # Plot heatmap data
    for ind_r, row in enumerate(row_keys):
        for ind_c, col in enumerate(col_keys):

            tup = heatmap(
                convert_to_img(heatmap_data_grid[ind_r][ind_c], [(0,0), (conv_min, conv_max), (acc_min, acc_max)], active_channels=[1, 2]), # normalize and convert to img
                ax=ax_lst[ind_r][ind_c],
                row_label=outer_grid_row_labels[ind_r],
                col_label=outer_grid_col_labels[ind_c],
                xticks=sp_range,
                yticks=dfr_range,
                activate_outer_grid_xlabel=True if ind_r == len(outer_grid_row_labels) - 1 else False,
                activate_outer_grid_ylabel=True if ind_c == 0 else False
            )

    # Add inner grid labels
    ax_lst[-1][-1].text(20.0, 19.5, "Sensor probability\nP(b|b) = P(w|w)", fontsize=15) # sensor probability as x label
    ax_lst[0][0].text(-5, -2, "On-fire tile fill ratio", fontsize=15) # fill ratio as y label

    # Add color legend
    color_leg_ax = right_gs_group.subplots(subplot_kw={"aspect": 40.0})
    color_leg_px_count = (60, 240)
    r = np.zeros((color_leg_px_count[1], color_leg_px_count[0]))
    g = np.repeat( np.reshape( np.linspace(1.0, 0.0, color_leg_px_count[1]), (color_leg_px_count[1], 1) ), color_leg_px_count[0], axis=1 )
    b = np.repeat( np.reshape( np.linspace(0.0, 1.0, color_leg_px_count[0]), (1, color_leg_px_count[0]) ), color_leg_px_count[1], axis=0)

    # Add color legend labels
    color_leg_ax.imshow( np.moveaxis( np.array( [r, g, b] ), 0, 2) )
    color_leg_ax.set_ylabel("Convergence", fontsize=15)
    color_leg_ax.set_xlabel("Accuracy", fontsize=15)

    # Add label limits for convergence
    color_leg_ax.text(-0.1*color_leg_px_count[0], 0.995*color_leg_px_count[1], "Slow", fontsize=12, rotation=90)
    color_leg_ax.text(-0.1*color_leg_px_count[0], 0.1*color_leg_px_count[0], "Fast", fontsize=12, rotation=90)

    # Add label limits for accuracy
    color_leg_ax.text(-0.01*color_leg_px_count[0], 1.02*color_leg_px_count[1], "Low", fontsize=12)
    color_leg_ax.text(0.86*color_leg_px_count[0], 1.02*color_leg_px_count[1], "High", fontsize=12)

    # Remove color legend
    color_leg_ax.set_xticks([])
    color_leg_ax.set_yticks([])

    # Add color bar
    # cbar_ax = right_gs_group.subplots(subplot_kw={"aspect": 15.0}) # add subplot with aspect ratio of 15
    # cbar = plt.colorbar(tup[2], cax=cbar_ax)
    # cbar.ax.set_ylabel("Convergence timestep (# of observations)", fontsize=15) # TODO: need to have a general version

    # Save the heatmap
    fig.set_size_inches(*fig_size)
    fig.savefig("/home/khaiyichin/heatmap.png", bbox_inches="tight", dpi=300)

# def generate_convergence_heatmap_data(v: VisualizationData, threshold: float):
#     """Generate the heatmap data using convergence values of the informed estimates.
#     """

#     return np.asarray(
#         [ [ v.detect_convergence(dfr, sp, threshold)[2]*v.comms_period for sp in v.sp_range ] for dfr in v.dfr_range ]
#     )

def generate_combined_heatmap_data(v: VisualizationData, threshold: float, order=(None, "convergence", "accuracy")):
    """

    Returns:
        A 3-D ndarray (containing 3 2-D arrays), with the 1st as the convergence heatmap and the 3rd
            as the accuracy heatmap. The 2nd array is populated with zeros.
    """

    output = np.zeros(shape=(3, len(v.dfr_range), len(v.sp_range)))
    conv_layer = []
    acc_layer = []

    for tfr in v.dfr_range:
        conv_layer_row = []
        acc_layer_row = []

        for sp in v.sp_range:

            tup = v.get_informed_estimate_metrics(tfr, sp, threshold)
            conv_layer_row.append(tup[0])
            acc_layer_row.append(tup[1])

        conv_layer.append(conv_layer_row)
        acc_layer.append(acc_layer_row)

    output[order.index("convergence")] = np.asarray(conv_layer)
    output[order.index("accuracy")] = np.asarray(acc_layer)

    return output

def convert_to_img(heatmap_ndarr: np.ndarray, limits: list, active_channels = [0,1,2]):
    """Convert the convergence and accuracy ndarray into a 3 channel image-type ndarray.

    The output would be a heatmap of the same size, but with values ranging from 0.0 to 1.0,
    with 0.0 meaning slow convergence/low accuracy and 1.0 meaning fast convergence/high accuracy.

    Args:
        heatmap_ndarr: A 3-D numpy array, with the 1st axis indicating the different data groups.
            E.g. heatmap_ndarr[0] = 2-D heatmap of first group, heatmap_ndarr[1] = 2-D array of second group.
        conv_lims: A tuple containing (min, max) convergence values.
        acc_lims: A tuple containing (min, max) convergence values.
    Returns:
        The same heatmap normalized and axis-flipped to an image with (m,n,3) dimensions.
    """

    def normalize(arr, min_val, max_val):
        arr_range = max_val - min_val
        return ( arr_range - ( arr - min_val ) ) / arr_range

    if 0 in active_channels:
        channel_0 = normalize(heatmap_ndarr[0], limits[0][0], limits[0][1])
        assert (np.all(channel_0 <= 1.0) and np.all(channel_0 >= 0.0)), "Normalized channel 0 values exceed [0.0, 1.0] range"
    else: channel_0 = np.zeros(heatmap_ndarr[0].shape)

    if 1 in active_channels:
        channel_1 = normalize(heatmap_ndarr[1], limits[1][0], limits[1][1])
        assert (np.all(channel_1 <= 1.0) and np.all(channel_1 >= 0.0)), "Normalized channel 1 values exceed [0.0, 1.0] range"
    else: channel_1 = np.zeros(heatmap_ndarr[1].shape)

    if 2 in active_channels:
        channel_2 = normalize(heatmap_ndarr[2], limits[2][0], limits[2][1])
        assert (np.all(channel_2 <= 1.0) and np.all(channel_2 >= 0.0)), "Normalized channel 2 values exceed [0.0, 1.0] range"
    else: channel_2 = np.zeros(heatmap_ndarr[2].shape)

    return np.moveaxis(np.array([channel_0, channel_1, channel_2]), 0, 2)

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
    # im = ax.imshow(heatmap_data, norm=LogNorm(vmin=kwargs["vmin"], vmax=kwargs["vmax"]), cmap="jet")
    im = ax.imshow(heatmap_data)

    # Show all ticks and label them with the respective list entries
    if kwargs["activate_outer_grid_xlabel"]:
        ax.set_xlabel(col_label, fontsize=15)
        ax.xaxis.labelpad = 25

    if kwargs["activate_outer_grid_ylabel"]:
        ax.set_ylabel(row_label, fontsize=15)
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