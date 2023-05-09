"""Visualization module
"""
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
import warnings
from joblib import Parallel, delayed
from abc import ABC, abstractmethod
import argparse
import timeit

from .sim_modules import ExperimentData, Sim
from pb2 import simulation_set_pb2

# Default values
CONV_THRESH = 5e-3
FIG_SIZE = (16, 12)
ACC_ABS_MAX = 0.25 # maximum accuracy threshold to draw in heatmap
XMIN_SCATTER = -0.01 # default minimum x-limit for the scatter plots
XMAX_SCATTER = 1.01 # default maximum x-limit for the scatter plots
YMIN_SCATTER = -0.01 # default minimum y-limit for the scatter plots
YMAX_SCATTER = 0.36 # default maximum y-limit for the scatter plots
YMIN_DECISION = 0.15 # default minimum y-limit for the decision plots
YMAX_DECISION = 1.1 # default maximum y-limit for the decision plots

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
        - number of trials,
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
                if os.path.splitext(f)[1] == ".ped" or os.path.splitext(f)[1] == ".pbs":
                    exp_data_obj_paths.append( os.path.join(root, f) )

        # Load the files
        for path in exp_data_obj_paths:

            # Check type of file to load
            if os.path.splitext(path)[1] == ".ped":
                print("Loading", path)
                obj = self.load_pkl_file(path)
            elif os.path.splitext(path)[1] == ".pbs":
                print("Loading", path)
                obj = self.load_proto_file(path)
            else: raise RuntimeError("Unknown extension encountered; please provide \".ped\" or \".pbs\" files.")

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
                self.num_trials = obj.num_trials
                self.num_steps = obj.num_steps
                self.tfr_range = obj.tfr_range
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
                similarity_bool = similarity_bool and (self.num_trials == obj.num_trials)
                similarity_bool = similarity_bool and (self.num_steps == obj.num_steps)

                # Check to see if either target fill ratio range or sensor probability range matches
                if (self.sp_range == obj.sp_range): # sensor probability matches

                    # Update the dictionary of tfr-sp_dict key-value pair
                    self.tfr_range.extend(obj.tfr_range)
                    self.stats_obj_dict.update(obj.stats_obj_dict)

                elif (self.tfr_range == obj.tfr_range): # target fill ratio matches

                    # Update the internal dictionary of sp-stats key-value pair
                    self.sp_range.extend(obj.sp_range)
                    [self.stats_obj_dict[i].update(obj.stats_obj_dict[i]) for i in obj.tfr_range]

                else: similarity_bool = False

                # Ensure that all the high level parameters are the same
                if not similarity_bool:
                    raise RuntimeError("The objects in the \"{0}\" directory do not have the same parameters.".format(exp_data_obj_paths))

        # Check whether the class was instantiated as an empty container or with actual paths
        if exp_data_obj_folder:
            self.tfr_range = sorted(self.tfr_range)
            self.sp_range = sorted(self.sp_range)

            self.aggregate_statistics()

    def load_pkl_file(self, folder_path, debug_data=False): return ExperimentData.load(folder_path, debug_data)

    def load_proto_file(self, folder_path):
        """Load SimulationStatsSet protobuf file into VisualizationData object.
        """
        with open(folder_path, "rb") as fopen:
            sim_stats_set_msg = simulation_set_pb2.SimulationStatsSet()
            sim_stats_set_msg.ParseFromString(fopen.read())

        # Add dynamic class member variables (remove abstraction layers)
        setattr(simulation_set_pb2.SimulationStatsSet, "sim_type", sim_stats_set_msg.sim_set.sim_type)
        setattr(simulation_set_pb2.SimulationStatsSet, "num_agents", sim_stats_set_msg.sim_set.num_agents)
        setattr(simulation_set_pb2.SimulationStatsSet, "num_trials", sim_stats_set_msg.sim_set.num_trials)
        setattr(simulation_set_pb2.SimulationStatsSet, "tfr_range", np.round(sim_stats_set_msg.sim_set.tfr_range, 3).tolist())
        setattr(simulation_set_pb2.SimulationStatsSet, "sp_range", np.round(sim_stats_set_msg.sim_set.sp_range, 3).tolist())
        setattr(simulation_set_pb2.SimulationStatsSet, "num_steps", sim_stats_set_msg.sim_set.num_steps)
        setattr(simulation_set_pb2.SimulationStatsSet, "comms_range", sim_stats_set_msg.sim_set.comms_range)
        setattr(simulation_set_pb2.SimulationStatsSet, "speed", np.round(sim_stats_set_msg.sim_set.speed, 3))
        setattr(simulation_set_pb2.SimulationStatsSet, "density", np.round(sim_stats_set_msg.sim_set.density, 3))
        setattr(simulation_set_pb2.SimulationStatsSet, "stats_obj_dict", {i: {j: None for j in sim_stats_set_msg.sp_range} for i in sim_stats_set_msg.tfr_range})

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

            # Extract x and conf and sp_mean_vals
            stats_obj.sp_distributed_sample_mean = stats_packet.rts.sp_mean_vals # why mean vals, not just one value?
            agent_informed_est = [[a.x for a in siv.agent_informed_vals] for siv in stats_packet.rts.swarm_informed_vals]
            agent_informed_conf = [[a.conf for a in siv.agent_informed_vals] for siv in stats_packet.rts.swarm_informed_vals]

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

            stats_obj.sp_distributed_sample_mean = np.asarray(stats_obj.sp_distributed_sample_mean)
            stats_obj.x = np.asarray(agent_informed_est) # (num_trials, num_agents, num_steps) ndarray
            stats_obj.gamma = np.asarray(agent_informed_conf) # (num_trials, num_agents, num_steps) ndarray

            sim_stats_set_msg.stats_obj_dict[np.round(stats_packet.packet.tfr, 3)][np.round(stats_packet.packet.b_prob, 3)] = stats_obj

        return sim_stats_set_msg

    def aggregate_statistics(self):
        """Aggregate the statistics.

        The estimates (and corresponding std. devs.) from the experiments within each simulation parameter setting
        will be combined to form one mean estimate (std. devs).
        """

        self.agg_stats_dict = {}

        for tfr_key, sp_dict in self.stats_obj_dict.items():

            temp_dict = {}

            for sp_key, stats_obj in sp_dict.items():

                # Compute mean across all trials
                x_hat_mean = np.mean(stats_obj.x_hat_sample_mean, axis=0)
                x_bar_mean = np.mean(stats_obj.x_bar_sample_mean, axis=0)
                x_mean = np.mean(stats_obj.x_sample_mean, axis=0)

                alpha_mean = np.mean(stats_obj.alpha_sample_mean, axis=0)
                rho_mean = np.mean(stats_obj.rho_sample_mean, axis=0)
                gamma_mean = np.mean(stats_obj.gamma_sample_mean, axis=0)

                # Compute pooled variance across all trials
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

            self.agg_stats_dict[tfr_key] = temp_dict

    # @todo modify this method to compute based on input of informed curve(s), not to compute all values!
    def detect_convergence(
        self,
        target_fill_ratio: float,
        sensor_prob: float,
        threshold=CONV_THRESH,
        aggregate=False,
        individual=False
    ):
        """Compute the point in time when convergence is achieved.

        This computes the convergence timestep (# of observations) for the local, social,
        and informed estimates. If the returned value is equal to the number of observations,
        that means convergence was not achieved.

        Args:
            target_fill_ratio: The target fill ratio used in the simulation.
            sensor_prob: The sensor probability used in the simulation.
            threshold: A float parametrizing the difference threshold.
            aggregate: Flag to choose whether to compute for aggregate data.

        Returns:
            The 3 indices at which convergence criterion is achieved. If `aggregate` == True, each of the 3
            outputs is a scalar; otherwise, they are lists of indices.
        """
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

        # Check type of curve to compute
        if aggregate and not individual:
            curves = [
                self.agg_stats_dict[target_fill_ratio][sensor_prob].x_hat_mean, # length of self.num_steps + 1
                self.agg_stats_dict[target_fill_ratio][sensor_prob].x_bar_mean, # length of self.num_steps / self.comms_period + 1
                self.agg_stats_dict[target_fill_ratio][sensor_prob].x_mean, # length of self.num_steps + 1
            ]

            # Go through all the curves
            output = Parallel(n_jobs=3, verbose=0)(delayed(parallel_inner_loop)(c) for c in curves) # a list of 3 variables

            assert len(output) == 3
            assert not isinstance(output[0], list)
            assert not isinstance(output[1], list)
            assert not isinstance(output[2], list)

        elif individual and not aggregate: # @todo: only computing for informed estimate!

            # Split the curve by trials
            # curves is an array of num_trials elements; each element in curves is a (num_agents, num_steps+1) ndarray
            curves = [*self.stats_obj_dict[target_fill_ratio][sensor_prob].x]

            # Go through all the curves
            conv_ind = [None for _ in range(self.num_trials)] # ends up being (num_trials, num_agents, 1) size

            for ind, agt_curves in enumerate(curves):
                conv_ind[ind] = Parallel(n_jobs=-1, verbose=0)(delayed(parallel_inner_loop)(c) for c in agt_curves) # agt_curves has shape (num_steps+1, 1)

            output = ([], [], conv_ind) # a tuple containing only the convergence indices of the informed estimates of all agents in all trials

            assert len(output) == 3
            assert len(output[2]) == self.num_trials
            assert len(output[2][0]) == self.num_agents

        elif not aggregate and not individual: # not aggregate and not individual so num_trials curves
            curves = [
                self.stats_obj_dict[target_fill_ratio][sensor_prob].x_hat_sample_mean, # (num_trials, num_steps+1) ndarray
                self.stats_obj_dict[target_fill_ratio][sensor_prob].x_bar_sample_mean, # (num_trials, num_steps+1) ndarray
                self.stats_obj_dict[target_fill_ratio][sensor_prob].x_sample_mean, # (num_trials, num_steps+1) ndarray
            ]

            # Go through all the curves
            conv_x_hat = Parallel(n_jobs=5, verbose=0)(delayed(parallel_inner_loop)(c) for c in curves[0])
            conv_x_bar = Parallel(n_jobs=5, verbose=0)(delayed(parallel_inner_loop)(c) for c in curves[1])
            conv_x = Parallel(n_jobs=5, verbose=0)(delayed(parallel_inner_loop)(c) for c in curves[2])

            output = (conv_x_hat, conv_x_bar, conv_x)

            assert len(output) == 3
            assert len(output[0]) == len(output[1]) and len(output[1]) == len(output[2])
            assert len(output[0]) == self.num_trials

        return output

    # @todo: modify this equation to compute accuracy based on input curve(s)
    def compute_accuracy(
        self,
        target_fill_ratio: float,
        sensor_prob: float,
        conv_ind_lst=None,
        aggregate=False,
        individual=False
    ):
        """Compute the estimate accuracy with respect to the target fill ratio.

        If the list of convergence indices is not provided, then convergence indices will be
        found in this function to use as target values to compute accuracies, based on the
        default convergence threshold.

        Args:
            target_fill_ratio: The target fill ratio used in the simulation.
            sensor_prob: The sensor probability used in the simulation.
            conv_ind_lst: The list of 3 convergence indices.
            aggregate: Flag to choose whether to compute for aggregate data.

        Returns:
            The 3 accuracies. If `aggregate` == True, each of the 3 outputs is a scalar; otherwise,
            they are lists of accuracies.
        """

        if conv_ind_lst is None:
            conv_ind_lst = self.detect_convergence(target_fill_ratio, sensor_prob, aggregate, individual)

        if aggregate and not individual:
            # conv_ind_lst has the shape (3, 1)

            acc_x_hat = abs(self.agg_stats_dict[target_fill_ratio][sensor_prob].x_hat_mean[conv_ind_lst[0]] - target_fill_ratio)
            acc_x_bar = abs(self.agg_stats_dict[target_fill_ratio][sensor_prob].x_bar_mean[conv_ind_lst[1]] - target_fill_ratio)
            acc_x = abs(self.agg_stats_dict[target_fill_ratio][sensor_prob].x_mean[conv_ind_lst[2]] - target_fill_ratio)

            output = ([acc_x_hat], [acc_x_bar], [acc_x])

            assert len(output) == 3
            assert len(output[0]) == len(output[1]) and len(output[1]) == len(output[2])
            assert len(output[0]) == self.num_trials

        elif individual and not aggregate: # only interested in informed estimates
            # conv_ind_lst has the form ([], [], index_lst) where index_lst has the shape (num_trials, num_agents)

            acc_x = []
            for trial_ind in range(self.num_trials):
                err = [
                    abs(self.stats_obj_dict[target_fill_ratio][sensor_prob].x[trial_ind][agt_ind][agt_conv_ind] - target_fill_ratio)
                            for agt_ind, agt_conv_ind in enumerate(conv_ind_lst[2][trial_ind])
                ] # err is a list with num_agents elements
                acc_x.append(err)

            output = ([], [], acc_x) # to maintain output consistency

            assert len(output[2]) == self.num_trials
            assert len(output[2][0]) == self.num_agents

        else:
            # conv_ind_lst has the shape (3, num_trials)

            acc_x_hat = [
                abs(self.stats_obj_dict[target_fill_ratio][sensor_prob].x_hat_sample_mean[trial_ind][conv_ind] - target_fill_ratio)
                for trial_ind, conv_ind in enumerate(conv_ind_lst[0])
            ]
            acc_x_bar = [
                abs(self.stats_obj_dict[target_fill_ratio][sensor_prob].x_bar_sample_mean[trial_ind][conv_ind] - target_fill_ratio)
                for trial_ind, conv_ind in enumerate(conv_ind_lst[1])
            ]
            acc_x = [
                abs(self.stats_obj_dict[target_fill_ratio][sensor_prob].x_sample_mean[trial_ind][conv_ind] - target_fill_ratio)
                for trial_ind, conv_ind in enumerate(conv_ind_lst[2])
            ]

            output = (acc_x_hat, acc_x_bar, acc_x)

            assert len(output) == 3
            assert len(output[0]) == len(output[1]) and len(output[1]) == len(output[2])
            assert len(output[0]) == self.num_trials

        return output

    def get_informed_estimate_metrics(self, target_fill_ratio: float, sensor_prob: float, threshold=CONV_THRESH, aggregate=False):
        """Get the accuracy and convergence of informed estimates.

        Args:
            target_fill_ratio: The target fill ratio used in the simulations.
            sensor_prob: The sensor probability used in the simulation.
            threshold: A float parametrizing the difference threshold.
            aggregate: Flag to choose whether to compute for aggregate data.

        Returns:
            A tuple of convergence (taking communication period into account) and accuracy values for the informed estimate.
        """

        # conv_lst = self.detect_convergence(curve) @todo abstract implementation of detect_convergence
        conv_lst = self.detect_convergence(target_fill_ratio, sensor_prob, threshold, aggregate)
        acc_lst = self.compute_accuracy(target_fill_ratio, sensor_prob, conv_lst, aggregate)

        conv_output = [conv_lst[2]] if aggregate else [i for i in conv_lst[2]]
        acc_output = [i for i in acc_lst[2]]

        return conv_output, acc_output

    def get_individual_informed_estimate_metrics(self, target_fill_ratio: float, sensor_prob: float, threshold=CONV_THRESH):

        conv_lst = self.detect_convergence(target_fill_ratio, sensor_prob, threshold, aggregate=False, individual=True)
        acc_lst = self.compute_accuracy(target_fill_ratio, sensor_prob, conv_lst, False, True)

        conv_output = [[agt_conv_ind for agt_conv_ind in trial] for trial in conv_lst[2]] # (num_trials, num_agents, 1) size
        acc_output = acc_lst[2] # should end up as (num_trials, num_agents, 1) size

        return conv_output, acc_output

    def get_decision_fractions(self, target_fill_ratio: float, sensor_prob: float or list, sim_step: int, bins=2):
        """Evaluate the decisions of each agent based on its informed estimate.

        Returns:
            A dict containing the fraction of correct decisions for each sensor probability.
        """

        # Create bins for the estimates
        bin_arr = np.linspace(0.0, 1.0, bins+1) # linspace gives the "checkpoints" for the bins, thus the bins are actually between each values

        # Define correct bin decision
        correct_bin = np.digitize(target_fill_ratio, bin_arr) # the output is between 1 to the number of bins, i.e., 1st bin, 2nd bin, etc.

        output_dict = {}

        # Process decision for each sensor probability
        for sp in sensor_prob:

            # Convert estimate to decision based on the number of bins
            estimates = self.stats_obj_dict[target_fill_ratio][sp].x

            decisions = np.take(estimates, sim_step, axis=2).flatten() # sim_step is the index to take from, thus has considered the initial estimate at t=0 already

            # Sort the decisions into bins
            binned_decisions = np.digitize(decisions, bin_arr)

            assert len(binned_decisions) == self.num_agents*self.num_trials

            # Count the fraction of correct decisions
            fraction = np.count_nonzero(binned_decisions == correct_bin) / len(binned_decisions)

            output_dict[sp] = fraction

        return output_dict

    def save(self, filepath=None, curr_time=None):
        """Serialize the class into a pickle.
        """

        # Get current time
        if curr_time is None:
            curr_time = datetime.now().strftime("%m%d%y_%H%M%S")

        if filepath:
            save_path = filepath + "_" + curr_time + ".vd"
        else:
            save_path = "vd_" + curr_time + ".vd"

        with open(save_path, "wb") as fopen:
            pickle.dump(self, fopen, pickle.HIGHEST_PROTOCOL)

        print("\nSaved VisualizationData object at: {0}.\n".format( os.path.abspath(save_path) ) )

    @classmethod
    def load(cls, filepath):
        """Load pickled data.
        """

        with open(filepath, "rb") as fopen:
            obj = pickle.load(fopen)

        # Verify the unpickled object
        assert isinstance(obj, cls)

        return obj

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
        print("\nLooking for VDG files in", self.folders, "\n")

    @abstractmethod
    def get_viz_data_obj(self, args: dict) -> VisualizationData:
        """Get VisualizationData object.
        """
        raise NotImplementedError("get_viz_data_obj function not implemented.")

    def get_tfr_range(self):

        # Get the first VisualizationData object
        viz_data_obj = list(self.viz_data_obj_dict.values())[0]

        while not isinstance(viz_data_obj, VisualizationData):
            viz_data_obj = list(viz_data_obj.values())[0]

        return viz_data_obj.tfr_range

    def get_sp_range(self):

        # Get the first VisualizationData object
        viz_data_obj = list(self.viz_data_obj_dict.values())[0]

        while not isinstance(viz_data_obj, VisualizationData):
            viz_data_obj = list(viz_data_obj.values())[0]

        return viz_data_obj.sp_range

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
        - number of trials,
    and with varying:
        - communication period,
        - communication probability,
        - number of agents.
    The target fill ratios and sensor probabilities are already varied (with fixed ranges) in the
    stored VisualizationData objects.

    This class is initialized with 1 argument:
        data_folder: A string specifying the directory containing all the ExperimentData files.
    """

    def __init__(self, data_folder=""):

        super().__init__(data_folder, ".ped")

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

        return self.viz_data_obj_dict[ args["comms_period"] ][ args["comms_prob"] ][ args["num_agents"] ]

    def save(self, filepath=None, curr_time=None):
        """Serialize the class into a pickle.
        """

        # Get current time
        if curr_time is None:
            curr_time = datetime.now().strftime("%m%d%y_%H%M%S")

        if filepath:
            save_path = filepath + "_" + curr_time + ".vdg"
        else:
            save_path = "vdg_static_" + curr_time + ".vdg"

        with open(save_path, "wb") as fopen:
            pickle.dump(self, fopen, pickle.HIGHEST_PROTOCOL)

        print( "\nSaved VisualizationDataGroupStatic object containing {0} items at: {1}.\n".format( self.stored_obj_counter, os.path.abspath(save_path) ) )

class VisualizationDataGroupDynamic(VisualizationDataGroupBase):
    """Class to store VisualizationData objects for dynamic simulations.

    The VisualizationData objects are stored by first using the VisualizationData's method
    in loading serialized SimulationStatSet protobuf files. Then each VisualizationData object
    are stored in this class.

    This class is intended to store VisualizationData objects with the same:
        - number of trials,
        - number of agents,
    and with varying:
        - robot speed, and
        - swarm density.
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

            if v.density in self.viz_data_obj_dict[v.speed]:
                raise ValueError("The data for speed={0}, density={1} exists already!".format(v.speed, v.density))
            else:
                self.viz_data_obj_dict[v.speed][v.density] = v
                self.stored_obj_counter += 1

    def get_viz_data_obj(self, args: dict) -> VisualizationData:
        """Get the VisualizationData object.

        Args:
            num_agents: The number of agents.
            speed: The robot speed.

        Returns:
            A VisualizationData object for the specified inputs.
        """

        return self.viz_data_obj_dict[ args["speed"] ][ args["density"] ]

    def save(self, filepath=None, curr_time=None):
        """Serialize the class into a pickle.
        """

        # Get current time
        if curr_time is None:
            curr_time = datetime.now().strftime("%m%d%y_%H%M%S")

        if filepath:
            save_path = filepath + "_" + curr_time + ".vdg"
        else:
            save_path = "vdg_dynamic_" + curr_time + ".vdg"

        with open(save_path, "wb") as fopen:
            pickle.dump(self, fopen, pickle.HIGHEST_PROTOCOL)

        print( "\nSaved VisualizationDataGroupDynamic object containing {0} items at: {1}.\n".format( self.stored_obj_counter, os.path.abspath(save_path) ) )

def plot_heatmap_vd(data_obj: VisualizationData, threshold: float, **kwargs):
    """Plot single heatmap based on a VisualizationData object. TODO: BROKEN, NEEDS FIXING
    """

    fig_size = FIG_SIZE
    fig = plt.figure(tight_layout=True, figsize=fig_size, dpi=175)

    # Create two groups: left for all the heatmaps, right for the color bar
    top_lvl_gs = fig.add_gridspec(1, 2, width_ratios=[10, 2.0])
    left_gs_group = top_lvl_gs[0].subgridspec(nrows=1, ncols=1)
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
    tfr_range = data_obj.tfr_range
    sp_range = data_obj.sp_range

    tup = heatmap(
        convert_to_img(heatmap_data, [(0,0), (0, data_obj.num_steps), (0.0, ACC_ABS_MAX)], active_channels=[1, 2]), # normalize and convert to img
        ax=ax,
        row_label="Black tile fill ratio",
        col_label="Sensor accuracy",
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
    color_leg_ax.imshow(np.moveaxis(np.array([r, g, b]), 0, 2))
    color_leg_ax.set_ylabel("Convergence", fontsize=15)
    color_leg_ax.set_xlabel("Accuracy", fontsize=15)

    # Add label limits for convergence
    color_leg_ax.text(
        -0.2 * color_leg_px_count[0],
        color_leg_px_count[1] - 0.5,
        "Slow",
        fontsize=10,
        rotation=90,
        ha="center",
        va="bottom"
    )
    color_leg_ax.text(
        -0.2 * color_leg_px_count[0], -0.5, "Fast", fontsize=10, rotation=90, ha="center", va="top"
    )
    # color_leg_ax.text(-0.1*color_leg_px_count[0], 0.995*color_leg_px_count[1], "Slow", fontsize=12, rotation=90)
    # color_leg_ax.text(-0.1*color_leg_px_count[0], 0.1*color_leg_px_count[0], "Fast", fontsize=12, rotation=90)

    # Add label ticks
    color_leg_ax.set_xticks(
        [-0.5, color_leg_px_count[0] - 0.5], ["Low\n({0})".format(1 - ACC_ABS_MAX), "High\n(1.0)"],
        fontsize=10
    )
    color_leg_ax.get_xticklabels()[0].set_ha("left")
    color_leg_ax.get_xticklabels()[-1].set_ha("right")
    color_leg_ax.set_yticks(
        [-0.5, color_leg_px_count[1] - 0.5], ["(0)", "({0})".format(data_obj.num_steps)],
        rotation=90,
        fontsize=10
    )
    color_leg_ax.get_yticklabels()[0].set_va("top")
    color_leg_ax.get_yticklabels()[-1].set_va("bottom")

    # Add title
    if "title" in kwargs: fig.suptitle(kwargs["title"], fontsize=20)

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

    # Create two groups: left for all the heatmaps, right for the color bar
    top_gs = fig.add_gridspec(1, 2, width_ratios=[10.0, 2.0], wspace=0.25)

    left_gs_group = top_gs[0].subgridspec(nrows=len(outer_grid_row_labels), ncols=len(outer_grid_col_labels), wspace=0.05)
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
            matrix = generate_combined_heatmap_data(
                v, threshold, order=(None, "convergence", "accuracy")
            )
            heatmap_data_grid_row.append(matrix)

            conv_min = np.amin([conv_min, np.amin(matrix[1])])
            conv_max = np.amax([conv_max, np.amax(matrix[1])])
            acc_min = np.amin([acc_min, np.amin(matrix[2])])
            acc_max = np.amax([acc_max, np.amax(matrix[2])])

        heatmap_data_grid.append(heatmap_data_grid_row)

    print("Convergence timestep minimum: {0}, maximum: {1}".format(conv_min, conv_max))
    print("Accuracy error minimum: {0}, maximum: {1}".format(acc_min, acc_max))

    # Extract the inner grid ranges (simply taken from the last VisualizationData object)
    tfr_range = v.tfr_range
    sp_range = v.sp_range

    # Plot heatmap data
    for ind_r, row in enumerate(row_keys):
        for ind_c, col in enumerate(col_keys):

            tup = heatmap(
                convert_to_img(heatmap_data_grid[ind_r][ind_c], [(0,0), (0, v.num_steps), (0.0, ACC_ABS_MAX)], active_channels=[1, 2]), # normalize and convert to img
                ax=ax_lst[ind_r][ind_c],
                row_label=outer_grid_row_labels[ind_r],
                col_label=outer_grid_col_labels[ind_c],
                xticks=sp_range,
                yticks=tfr_range,
                activate_outer_grid_xlabel=True if ind_r == len(outer_grid_row_labels) - 1 else False,
                activate_outer_grid_ylabel=True if ind_c == 0 else False
            )

    # Add inner grid labels
    ax_lst[-1][-1].text(20.0, 20.0, "Sensor\naccuracy", fontsize=13) # sensor probability as x label
    ax_lst[0][0].text(-5, -2, "Black tile fill ratio", fontsize=13) # fill ratio as y label

    # Add color legend
    color_leg_ax = right_gs_group.subplots()
    color_leg_px_count = (70, 420)
    r = np.zeros( (color_leg_px_count[1], color_leg_px_count[0]) )
    g = np.repeat( np.reshape( np.linspace(1.0, 0.0, color_leg_px_count[1]), (color_leg_px_count[1], 1) ), color_leg_px_count[0], axis=1 )
    b = np.repeat( np.reshape( np.linspace(0.0, 1.0, color_leg_px_count[0]), (1, color_leg_px_count[0]) ), color_leg_px_count[1], axis=0 )

    # Add color legend labels
    color_leg_ax.imshow(np.moveaxis(np.array([r, g, b]), 0, 2))
    color_leg_ax.set_ylabel("Convergence\n(steps)", fontsize=13)
    color_leg_ax.set_xlabel("Accuracy\n(1 - abs. error)", fontsize=13)

    # Add label limits for convergence
    color_leg_ax.text(
        -0.2 * color_leg_px_count[0],
        color_leg_px_count[1] - 0.5,
        "Slow",
        fontsize=10,
        rotation=90,
        ha="center",
        va="bottom"
    )
    color_leg_ax.text(
        -0.2 * color_leg_px_count[0], -0.5, "Fast", fontsize=10, rotation=90, ha="center", va="top"
    )

    # Add label limits for accuracy
    # color_leg_ax.text(-0.5, 1.04*color_leg_px_count[1], "({0})".format(1 - ACC_ABS_MAX), fontsize=10, ha="left")
    # color_leg_ax.text(color_leg_px_count[0]-0.5, 1.04*color_leg_px_count[1], "(1.0)", fontsize=10, ha="right")

    # Add label ticks
    color_leg_ax.set_xticks([-0.5, color_leg_px_count[0]-0.5], ["Low\n({0})".format(1- ACC_ABS_MAX), "High\n(1.0)"], fontsize=10)
    color_leg_ax.get_xticklabels()[0].set_ha("left")
    color_leg_ax.get_xticklabels()[-1].set_ha("right")
    color_leg_ax.set_yticks([-0.5, color_leg_px_count[1]-0.5], ["(0)", "({0})".format(v.num_steps)], rotation=90, fontsize=10)
    color_leg_ax.get_yticklabels()[0].set_va("top")
    color_leg_ax.get_yticklabels()[-1].set_va("bottom")

    # Add title
    if "title" in kwargs: fig.suptitle(kwargs["title"], fontsize=20)

    # Save the heatmap
    fig.set_size_inches(*fig_size)
    if isinstance(data_obj, VisualizationDataGroupDynamic): data_obj_type = "dyn"
    else: data_obj_type = "sta"
    fig.savefig("/home/khaiyichin/heatmap_"+data_obj_type+"_"+"conv{0}_s{1}_t{2}".format(int(np.round(threshold*1e3 ,3)), v.num_steps, v.num_trials)+".png", bbox_inches="tight", dpi=300)


def generate_combined_heatmap_data(
    v: VisualizationData, threshold: float, order=(None, "convergence", "accuracy")
):
    """

    Returns:
        A 3-D ndarray (containing 3 2-D arrays), with the 1st as the convergence heatmap and the 3rd
            as the accuracy heatmap. The 2nd array is populated with zeros.
    """

    output = np.zeros(shape=(3, len(v.tfr_range), len(v.sp_range)))
    conv_layer = []
    acc_layer = []

    for tfr in v.tfr_range:
        conv_layer_row = []
        acc_layer_row = []

        for sp in v.sp_range:

            tup = v.get_informed_estimate_metrics(tfr, sp, threshold, True)
            conv_layer_row.append(*tup[0])
            acc_layer_row.append(*tup[1])

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

    if 0 in active_channels:
        channel_0 = normalize(heatmap_ndarr[0], limits[0][0], limits[0][1], True)
        assert (np.all(channel_0 <= 1.0) and np.all(channel_0 >= 0.0)), "Normalized channel 0 values exceed [0.0, 1.0] range"
    else: channel_0 = np.zeros(heatmap_ndarr[0].shape)

    if 1 in active_channels:
        channel_1 = normalize(heatmap_ndarr[1], limits[1][0], limits[1][1], True)
        assert (np.all(channel_1 <= 1.0) and np.all(channel_1 >= 0.0)), "Normalized channel 1 values exceed [0.0, 1.0] range"
    else: channel_1 = np.zeros(heatmap_ndarr[1].shape)

    if 2 in active_channels:
        channel_2 = normalize(heatmap_ndarr[2], limits[2][0], limits[2][1], True)
        assert (np.all(channel_2 <= 1.0) and np.all(channel_2 >= 0.0)), "Normalized channel 2 values exceed [0.0, 1.0] range"
    else: channel_2 = np.zeros(heatmap_ndarr[2].shape)

    return np.moveaxis(np.array([channel_0, channel_1, channel_2]), 0, 2)


def heatmap(
    heatmap_data,
    row_label="",
    col_label="",
    xticks=[],
    yticks=[],
    ax=None,
    cbar_kw={},
    cbarlabel="",
    valfmt="{x:.2f}",
    **kwargs
):
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
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")

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

    return fig, ax, im

def plot_scatter(data_obj: VisualizationData, threshold: float, args, individual=True, **kwargs):

    # Define variables depending on individual vs. average and static vs. dynamic
    if individual:
        informed_est_method = data_obj.get_individual_informed_estimate_metrics

    else:
        informed_est_method = data_obj.get_informed_estimate_metrics

    if args["data_type"] == "static":
        period = args["comms_period"]
        n = args["num_agents"]
        filename_param_1 = "prd{0}_cprob{1}_agt{2}".format(int(period), 1, int(n))
        data_obj_type = "sta"

    elif args["data_type"] == "dynamic":
        speed = args["speed"]
        density = args["density"]
        filename_param_1 = "spd{0}_den{1}".format(int(speed), int(density))
        data_obj_type = "dyn"

    tfr = args["tfr"]
    sp = args["sp"]

    # Create figures for plotting
    fig_size = (6, 4)
    fig, ax_lst = plt.subplots(
        1,
        2,
        tight_layout=True,
        figsize=fig_size,
        dpi=175,
        gridspec_kw={"width_ratios": [7, 1]}
    )

    # Get performance metrics for all trials
    conv_min = np.inf
    conv_max = -np.inf
    acc_min = np.inf
    acc_max = -np.inf

    if isinstance(tfr, list): # sp is fixed
        manipulated_var = tfr
        fixed = "sp"

    elif isinstance(sp, list): # tfr is fixed
        manipulated_var = sp
        fixed = "tfr"

    else: # both tfr and sp are scalar
        fixed = "tfr"
        manipulated_var = [sp]

    # Decode distribution values
    distributed_case = [ val for val in manipulated_var if val < 0.0 ]
    if distributed_case: id_offset = 0 # enable the black marker (which has index = 0)
    else: id_offset = 1 # black marker is reserved for the heterogeneous data

    # Create the scatter plots
    median_lst = []
    for ind, var in enumerate(manipulated_var):

        if fixed == "sp":
            conv_lst, acc_lst = informed_est_method(var, sp, threshold)
        elif fixed == "tfr":
            conv_lst, acc_lst = informed_est_method(tfr, var, threshold)

        # Compute minimum and maximum values
        conv_min = np.amin([conv_min, np.amin(conv_lst)])
        conv_max = np.amax([conv_max, np.amax(conv_lst)])
        acc_min = np.amin([acc_min, np.amin(acc_lst)])
        acc_max = np.amax([acc_max, np.amax(acc_lst)])

        # @todo: maybe give them individual marker types? for now just make all the ones in the same trial the same, i.e., flatten data
        if individual: conv_lst = [c for i in conv_lst for c in i]
        if individual: acc_lst = [c for i in acc_lst for c in i]

        # conv_norm = normalize(np.asarray(conv_lst), 0, data_obj.num_steps).tolist()
        # acc_norm = normalize(np.asarray(acc_lst), 0, ACC_ABS_MAX)
        comms_steps = convert_sim_steps_to_comms_rounds(np.asarray(conv_lst), data_obj.comms_period)

        # Calculate median
        median_lst.append( (np.median(comms_steps), np.median(acc_lst)) )
        scatter(
            [comms_steps, acc_lst],
            c=[ind + id_offset] * len(conv_lst),
            ax=ax_lst[0],
            vmin=0,
            vmax=len(manipulated_var) - 1 + id_offset,
            edgecolors="none",
            alpha=0.2,
            s=30
        ) # general transparent data

        # Modify the label for the distributed case
        if var < 0.0: manipulated_var[ind] = decode_sp_distribution_key(var)

    scatter(
        list(zip(*median_lst)),
        c=range(id_offset, len(median_lst) + id_offset),
        ax=ax_lst[0],
        vmin=0,
        vmax=len(manipulated_var) - 1 + id_offset,
        edgecolors="black",
        alpha=1.0,
        s=65
    ) # median points

    print("Convergence timestep minimum: {0}, maximum: {1}".format(conv_min, conv_max))
    print("Accuracy error minimum: {0}, maximum: {1}".format(acc_min, acc_max))

    # Set plot parameters
    if args["ymax"]: ymax = args["ymax"]
    else: ymax = YMAX_SCATTER
    max_comms_rounds = data_obj.num_steps/data_obj.comms_period # convert the x-axis to use communications rounds

    ax_lst[0].set_xlabel("Communication Rounds", fontsize=14)
    ax_lst[0].set_ylabel("Absolute Error", fontsize=14)
    ax_lst[0].set_xlim(-0.02 * max_comms_rounds, 1.02 * max_comms_rounds)
    ax_lst[0].set_ylim(YMIN_SCATTER, ymax)
    ax_lst[0].xaxis.set_tick_params(labelsize=10)
    ax_lst[0].yaxis.set_tick_params(labelsize=10)
    ax_lst[0].grid()

    # Create color bar
    color_bar_img = list(
        zip(*[list(range(id_offset, len(manipulated_var) + id_offset))])
    ) # size of len(manipulated_var) x 1 list
    ax_lst[1].imshow(
        color_bar_img,
        aspect=2,
        cmap="nipy_spectral",
        vmin=0,
        vmax=(len(manipulated_var) - 1 + id_offset)
    )

    # Modify tick labels
    ax_lst[1].yaxis.set_label_position("right")
    ax_lst[1].yaxis.tick_right()
    ax_lst[1].set_xticks([])
    ax_lst[1].set_yticks(list(range(len(manipulated_var))), manipulated_var, fontsize=8)

    if fixed == "tfr":
        ax_lst[1].set_ylabel("Sensor Accuracies", fontsize=14)
        filename_param_2 = "tfr{0}".format(int(tfr * 1e3))
    elif fixed == "sp":
        ax_lst[1].set_ylabel("Target fill ratios", fontsize=14)
        filename_param_2 = "tfr{0}".format(int(sp * 1e3))

    # Save the scatter plot
    filename_param = "{0}_{1}".format(filename_param_2, filename_param_1)

    fig.savefig(
        "scatter_" + data_obj_type + "_" + "conv{0}_s{1}_t{2}_{3}".format(
            int(np.round(threshold * 1e3, 3)),
            int(data_obj.num_steps),
            int(data_obj.num_trials),
            filename_param
        ) + ".png",
        bbox_inches="tight",
        dpi=300
    )

def scatter(scatter_data, ax=None, **kwargs):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax.scatter(scatter_data[0], scatter_data[1], cmap="nipy_spectral", **kwargs)

def plot_decision(data_obj: VisualizationData, args):

    # Assigned fixed colors to sensor probabilities; @TODO: kind of hacky
    fixed_sensor_probability_bins = np.linspace(0.5125, 0.9875, 20) # 19 bins

    tfr = args["tfr"]
    sp = args["sp"]
    bins = args["bins"]
    sim_steps = args["sim_steps"]

    if args["data_type"] == "static":
        period = args["comms_period"]
        n = args["num_agents"]
        filename_param_1 = "prd{0}_cprob{1}_agt{2}_bins{3}".format(int(period), 1, int(n), bins)
        data_obj_type = "sta"

    elif args["data_type"] == "dynamic":
        speed = args["speed"]
        density = args["density"]
        filename_param_1 = "spd{0}_den{1}_bins{2}".format(int(speed), int(density), bins)
        data_obj_type = "dyn"

    # Create the figures and axes
    fig_size = (6, 4)
    fig, ax_lst = plt.subplots(
        1,
        2,
        tight_layout=True,
        figsize=fig_size,
        dpi=175,
        gridspec_kw={"width_ratios": [6 * len(fixed_sensor_probability_bins) / len(sp), 1]})

    # Convert sensor probability values into IDs used for deciding marker colors (the colors are fixed for sensor probability values)
    id_lst = []

    for s in sp:
        if s >= 0.0: id_lst.append(np.digitize(s, fixed_sensor_probability_bins)) # homogeneous sensor probabilities
        else: id_lst.append(0) # the distributed sensor probability case uses the black marker

    # Define array of colors according to the nipy_spectral colormap
    c = plt.cm.nipy_spectral(np.array(id_lst) / (len(fixed_sensor_probability_bins) - 1))

    # Obtain the decision fractions
    decision_fractions = {step: data_obj.get_decision_fractions(tfr, sp, step, bins) for step in sim_steps}

    # Check the decision fraction of each sensor probability to find the ones that clutter close to 1 (using 0.9 as a threshold)
    cluttered_keys = set()
    for sp_dict in decision_fractions.values():
        cluttered_keys.update((k for k, v in sp_dict.items() if v > 0.9))

    # Add offsets to reduce clutter of points so that the markers/lines that overlap is still visible
    max_comms_rounds = data_obj.num_steps/data_obj.comms_period
    offset = (np.linspace(0, 1, len(cluttered_keys)) * max_comms_rounds/10 - max_comms_rounds/20).tolist()
    offset = [offset.pop(0) if s in cluttered_keys else 0.0 for _, s in enumerate(sp)]

    # Plot the lines for each sensor probability
    for ind, s in enumerate(sp):

        points = [decision_fractions[k][s] for k in decision_fractions.keys()]

        line(
            line_data=[np.array(sim_steps) / data_obj.comms_period + offset[ind], points],
            ax=ax_lst[0],
            ls="-", # line style
            lw=1, # line width
            marker="d", # marker type
            ms="15", # marker size
            mfc=c[ind], # marker face color
            c=c[ind] # color
        )

        # Add markers to those that have 100% correct decisions
        consensus_ind = [i for i, p in enumerate(points) if p == 1.0]
        if consensus_ind:
            line(
                line_data=[np.array(sim_steps[consensus_ind]) / data_obj.comms_period + offset[ind], [1.0] * len(consensus_ind)],
                ax=ax_lst[0],
                marker="|", # marker type
                ms="35", # marker size
                c=c[ind] # color
            )

    # Modify the label for the distributed case
    if 0 in id_lst:
        sp[id_lst.index(0)] = decode_sp_distribution_key(sp[id_lst.index(0)])

    # Add ticker formatter (mostly for the log scale that is unused for now, but may be helpful in the future)
    ticker_formatter = FuncFormatter(lambda y, _: "{:.4g}".format(y))

    ax_lst[0].set_xlabel("Communication Rounds", fontsize=14)
    ax_lst[0].set_ylabel("Fraction of Correct Decisions", fontsize=14)
    ax_lst[0].set_xticks(np.array(sim_steps) / data_obj.comms_period)
    # ax_lst[0].set_yscale("log")
    ax_lst[0].set_ylim(bottom=YMIN_DECISION, top=YMAX_DECISION)
    ax_lst[0].xaxis.set_tick_params(which="both", labelsize=10)
    ax_lst[0].yaxis.set_tick_params(which="both", labelsize=10)
    ax_lst[0].yaxis.grid(which="both", linestyle=":")
    ax_lst[0].xaxis.grid(which="major", linestyle=":")
    ax_lst[0].yaxis.set_major_formatter(ticker_formatter)
    ax_lst[0].yaxis.set_minor_formatter(ticker_formatter)

    # Create color bar
    color_bar_img = np.array(id_lst, ndmin=2).T
    ax_lst[1].imshow(
        color_bar_img,
        aspect=2,
        cmap="nipy_spectral",
        vmin=0,
        vmax=len(fixed_sensor_probability_bins) - 1
    )

    # Modify tick labels
    ax_lst[1].yaxis.set_label_position("right")
    ax_lst[1].yaxis.tick_right()
    ax_lst[1].set_xticks([])
    ax_lst[1].set_yticks(range(len(id_lst)), sp, fontsize=8)
    ax_lst[1].set_ylabel("Sensor Accuracies", fontsize=14)

    filename_param_2 = "tfr{0}".format(int(tfr * 1e3))

    # Save the decision plot
    filename_param = "{0}_{1}".format(filename_param_2, filename_param_1)

    fig.savefig(
        "decision_" + data_obj_type + "_" +
        "s{0}_t{1}_{2}".format(int(data_obj.num_steps), int(data_obj.num_trials), filename_param) +
        ".png",
        bbox_inches="tight",
        dpi=300
    )

def line(line_data, ax=None, **kwargs):
    if ax is None:
        raise NotImplementedError("Line drawing function is not generally available yet.")
    else:
        ax.plot(line_data[0], line_data[1], **kwargs)

def plot_timeseries(
    target_fill_ratio,
    sensor_prob,
    data_obj: VisualizationData,
    agg_data=False,
    convergence_thresh=CONV_THRESH
):
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
    fig_x_hat.set_size_inches(8, 6)

    fig_x_bar, ax_x_bar = plt.subplots(2, sharex=True)
    fig_x_bar.set_size_inches(8, 6)

    fig_x, ax_x = plt.subplots(2, sharex=True)
    fig_x.set_size_inches(8, 6)

    abscissa_values_x_hat = list(range(data_obj.num_steps + 1))
    abscissa_values_x_bar = list(
        range(0, data_obj.num_steps + 1 * data_obj.comms_period, data_obj.comms_period)
    )
    abscissa_values_x = list(range(data_obj.num_steps + 1))

    # Plot for all trials
    if not agg_data:

        # Compute the convergence timestamps
        conv_ind_x_hat, conv_ind_x_bar, conv_ind_x = \
            data_obj.detect_convergence(target_fill_ratio, sensor_prob, convergence_thresh, False, False)

        for i in range(data_obj.num_trials):
            stats_obj = data_obj.stats_obj_dict[target_fill_ratio][sensor_prob]

            # Plot time evolution of local estimates and confidences
            x_hat_bounds = compute_std_bounds(stats_obj.x_hat_sample_mean[i], stats_obj.x_hat_sample_std[i])
            alpha_bounds = compute_std_bounds(stats_obj.alpha_sample_mean[i], stats_obj.alpha_sample_std[i])

            traj_x_hat = ax_x_hat[0].plot(abscissa_values_x_hat, stats_obj.x_hat_sample_mean[i], label="Exp {}".format(i))
            ax_x_hat[0].axvline(abscissa_values_x_hat[conv_ind_x_hat[i]], color=traj_x_hat[0].get_c(), linestyle=":")
            ax_x_hat[0].axhline(target_fill_ratio, color="black", linestyle="--")
            ax_x_hat[0].fill_between(abscissa_values_x_hat, x_hat_bounds[0], x_hat_bounds[1], alpha=0.2)

            traj_x_hat_conf = ax_x_hat[1].plot(abscissa_values_x_hat, stats_obj.alpha_sample_mean[i], label="Exp {}".format(i))
            ax_x_hat[1].axvline(abscissa_values_x_hat[conv_ind_x_hat[i]], color=traj_x_hat_conf[0].get_c(), linestyle=":") # draw convergence line
            ax_x_hat[1].fill_between(abscissa_values_x_hat, alpha_bounds[0], alpha_bounds[1], alpha=0.2)

            # Plot time evolution of social estimates and confidences
            x_bar_bounds = compute_std_bounds(stats_obj.x_bar_sample_mean[i], stats_obj.x_bar_sample_std[i])
            rho_bounds = compute_std_bounds(stats_obj.rho_sample_mean[i], stats_obj.rho_sample_std[i])

            traj_x_bar = ax_x_bar[0].plot(abscissa_values_x_bar, stats_obj.x_bar_sample_mean[i], label="Exp {}".format(i))
            ax_x_bar[0].axvline(abscissa_values_x_bar[conv_ind_x_bar[i]], color=traj_x_bar[0].get_c(), linestyle=":")
            ax_x_bar[0].axhline(target_fill_ratio, color="black", linestyle="--")
            ax_x_bar[0].fill_between(abscissa_values_x_bar, x_bar_bounds[0], x_bar_bounds[1], alpha=0.2)

            traj_x_bar_conf = ax_x_bar[1].plot(abscissa_values_x_bar, stats_obj.rho_sample_mean[i], label="Exp {}".format(i))
            ax_x_bar[1].axvline(abscissa_values_x[conv_ind_x_bar[i]], color=traj_x_bar_conf[0].get_c(), linestyle=":") # draw convergence line
            ax_x_bar[1].fill_between(abscissa_values_x_bar, rho_bounds[0], rho_bounds[1], alpha=0.2)

            # Plot time evolution of informed estimates and confidences
            x_bounds = compute_std_bounds(stats_obj.x_sample_mean[i], stats_obj.x_sample_std[i])
            gamma_bounds = compute_std_bounds(stats_obj.gamma_sample_mean[i], stats_obj.gamma_sample_std[i])

            traj_x = ax_x[0].plot(abscissa_values_x, stats_obj.x_sample_mean[i], label="Exp {}".format(i))
            ax_x[0].axvline(abscissa_values_x[conv_ind_x[i]], color=traj_x[0].get_c(), linestyle=":")
            ax_x[0].axhline(target_fill_ratio, color="black", linestyle="--")
            ax_x[0].fill_between(abscissa_values_x, x_bounds[0], x_bounds[1], alpha=0.2)

            traj_x_conf = ax_x[1].plot(abscissa_values_x, stats_obj.gamma_sample_mean[i], label="Exp {}".format(i))
            ax_x[1].axvline(abscissa_values_x[conv_ind_x[i]], color=traj_x_conf[0].get_c(), linestyle=":") # draw convergence line
            ax_x[1].fill_between(abscissa_values_x, gamma_bounds[0], gamma_bounds[1], alpha=0.2)

    else:

        # Aggregate statistics
        agg_stats_obj = data_obj.agg_stats_dict[target_fill_ratio][sensor_prob]

        # Compute the convergence timestamps
        conv_ind_x_hat, conv_ind_x_bar, conv_ind_x = \
            data_obj.detect_convergence(target_fill_ratio, sensor_prob, convergence_thresh, True, False)

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

def plot_individual_timeseries(
    target_fill_ratio, sensor_prob, data_obj: VisualizationData, convergence_thresh=CONV_THRESH
):

    # Create figure and axes handles
    fig_lst = [None for i in range(data_obj.num_trials)]
    ax_lst = [None for i in range(data_obj.num_trials)]

    abscissa_values_x = list(range(data_obj.num_steps + 1))

    # Compute convergence for informed estimates
    conv_lst = data_obj.detect_convergence(target_fill_ratio, sensor_prob, convergence_thresh, False, True)[2]

    # Plot data
    for i in range(data_obj.num_trials):

        fig_lst[i], ax_lst[i] = plt.subplots(2, sharex=True)
        fig_lst[i].set_size_inches(8, 6)

        stats_obj = data_obj.stats_obj_dict[target_fill_ratio][sensor_prob]

        # Plot time evolution of informed estimates and confidences by iterating through agents
        for j in range(data_obj.num_agents):
            traj_x = ax_lst[i][0].plot(abscissa_values_x, stats_obj.x[i][j]) # draw estimate trajectory
            ax_lst[i][0].axvline(abscissa_values_x[conv_lst[i][j]], color=traj_x[0].get_c(), linestyle=":") # draw convergence line

            traj_conf = ax_lst[i][1].plot(abscissa_values_x, stats_obj.gamma[i][j]) # draw confidence trajectory
            ax_lst[i][0].axvline(abscissa_values_x[conv_lst[i][j]], color=traj_conf[0].get_c(), linestyle=":") # draw convergence line

        # Set axis properties
        ax_lst[i][0].set_title("{0} agents' informed values (fill ratio: {1}, sensor: {2})".format(data_obj.num_agents, target_fill_ratio, sensor_prob))
        ax_lst[i][0].set_ylabel("Informed estimates")
        ax_lst[i][0].set_ylim(0, 1.0)
        ax_lst[i][1].set_ylim(10e-3, 10e5)
        ax_lst[i][1].set_ylabel("Informed confidences")
        ax_lst[i][1].set_xlabel("Observations")
        ax_lst[i][1].set_yscale("log")

        # Turn on grid lines
        activate_subplot_grid_lines(ax_lst[i])

def decode_sp_distribution_key(encoded_val):

    assert encoded_val < 0.0

    encoded_val_str = str(-encoded_val)

    # Decode ID
    if encoded_val_str[0] == "2":
        id = "U"
    elif encoded_val_str[0] == "3":
        id = "N"

    # Decode parameters
    param_1 = float(encoded_val_str[1:5]) * 1e-3
    param_2 = float(encoded_val_str[5:]) * 1e-3

    return "{0}({1}, {2})".format(id, param_1, param_2)

def normalize(arr, min_val, max_val, flip=False):
    arr_range = max_val - min_val
    return (arr_range - (arr - min_val)) / arr_range if flip else (arr - min_val) / arr_range

def convert_sim_steps_to_comms_rounds(sim_steps_arr, comms_period):
    return sim_steps_arr / comms_period

def compute_std_bounds(mean_val, std_val):
    return [np.add(mean_val, std_val), np.subtract(mean_val, std_val)]

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
    subplot_fig.legend(
        handles, labels, loc="center right", bbox_to_anchor=(box_2.width * 1.25, 0.5)
    )

class Visualizer:

    def __init__(self, data_type):
        if data_type == "static":
            v_type_str = "ExperimentData pickle"
            u_types_str = "communications period, communication probability, and number of agents"

            self.data_type = "static"
            self.u_types_lst = ["comms_period", "comms_prob", "num_agents"]
            self.vdg_cls = VisualizationDataGroupStatic

        elif data_type == "dynamic":
            v_type_str = "SimulationStatsSet protobuf"
            u_types_str = "robot speed and swarm density"

            self.data_type = "dynamic"
            self.u_types_lst = ["speed", "density"]
            self.vdg_cls = VisualizationDataGroupDynamic
        else:
            raise ValueError("Unknown data type {0}!".format(data_type))

        # Create common arguments
        parser = argparse.ArgumentParser(description="Visualize {0} multi-agent simulation data".format(data_type))
        parser.add_argument("FILE", type=str, \
            help="path to folder containing serialized {0} files or path to a VisualizationDataGroup pickle file (see the \"-g\" flag)".format(v_type_str))
        parser.add_argument("CONV", type=float, \
            help="convergence threshold value")
        parser.add_argument("-g", action="store_true", \
            help="flag to indicate if the path is pointing to a VisualizationDataGroup pickle file")
        parser.add_argument("-a", action="store_true", \
            help="flag to use aggregate data instead of data from individual trials (exclusive with the \"-i\" flag)")
        parser.add_argument("-i", action="store_true", \
            help="flag to show individual agent data (only used for time series and scatter plot data with the \"-U\" flag; exclusive with the \"-a\" flag)")
        parser.add_argument("-s", action="store_true", \
            help="flag to show the plots")
        parser.add_argument("--steps", type=int, \
            help="first n simulation steps to evaluate to (default: evaluate from start to end of simulation)")

        # Add subparsing arguments
        viz_type_subparser = parser.add_subparsers(dest="viz_type", required=True, help ="commands for visualization type")
        series_subparser = viz_type_subparser.add_parser("series", help="visualize time series data")
        heatmap_subparser = viz_type_subparser.add_parser("heatmap", help="visualize heatmap data")
        scatter_subparser = viz_type_subparser.add_parser("scatter", help="visualize scatter data")
        decision_subparser = viz_type_subparser.add_parser("decision", help="visualize collective-decision making data")

        # Time series plot options
        series_subparser.add_argument("-TFR", required=True, type=float, help="single target fill ratio to use in plotting time series data")
        series_subparser.add_argument("-SP", required=True, type=float, help="single sensor probability to use in plotting time series data")
        series_subparser.add_argument("-U", required=True, nargs="*", type=float, help="{0} to use in plotting time series data".format(u_types_str))

        # Heatmap plot options
        heatmap_subparser.add_argument("-u", nargs="*", \
            help="(optional) {0} to use in plotting single heatmap data".format(u_types_str))
        heatmap_subparser.add_argument("-rstr", nargs="+", type=str, help="(optional) outer grid row labels (must match number of \"ROW\" arguments; unused if \"-u\" arguments are used)")
        heatmap_subparser.add_argument("-row", nargs="+", type=float, help="(optional) outer grid row coordinates (unused if \"-u\" arguments are used)")
        heatmap_subparser.add_argument("-cstr", nargs="+", type=str, help="(optional) outer grid column labels (must match number of \"COL\" arguments; unused if \"-u\" arguments are used)")
        heatmap_subparser.add_argument("-col", nargs="+", type=float, help="(optional) outer grid column coordinates (unused if \"-u\" arguments are used)")

        # Scatter plot options
        scatter_subparser.add_argument("-tfr", nargs="+", type=float, help="(optional) target fill ratio to use in plotting scatter data (must provide single \"-sp\" argument if this is not provided)")
        scatter_subparser.add_argument("-sp", nargs="+", type=float, help="(optional) sensor probability to use in plotting scatter data (must provide single \"-tfr\" argument if this is not provided)")
        scatter_subparser.add_argument("-U", required=True, nargs="*", type=float, help="{0} to use in plotting scatter data".format(u_types_str))
        scatter_subparser.add_argument("--ymax", default=None, type=float, help="(optional) maximum y limit value for the scatter plot")

        # Decision scatter plot options
        decision_subparser.add_argument("-TFR", required=True, type=float, help="single target fill ratio to use in plotting collective decision data")
        decision_subparser.add_argument("-sp", nargs="+", type=float, help="(optional) sensor probability to use in plotting collective decision data")
        decision_subparser.add_argument("-U", required=True, nargs="+", help="{0} to use in plotting collective decision data".format(u_types_str))
        decision_subparser.add_argument("--bins", type=int, default=2, help="(optional) the number of bins to separate the swarm's decision (default: 2)")
        decision_subparser.add_argument("--step_inc", type=int, default=1000, help="(optional) the increment in simulation steps to evaluate decisions (default: 1000)")

        args = parser.parse_args()

        self.load_data(args)

        start = timeit.default_timer()

        self.generate_plot(args)

        end = timeit.default_timer()

        print('Elapsed time:', end - start)

        if args.s:
            plt.show()

    def load_data(self, args):

        # Check whether to load VisualizationDataGroup object
        if args.g:
            self.data = self.vdg_cls.load(args.FILE)

            # Load specific VisualizationData object from VisualizationDataGroup object
            try:
                u_args = args.u
            except Exception as e:
                try:
                    u_args = args.U
                except Exception as e:
                    u_args = None

            if u_args:
                if (self.data_type == "static") and (len(u_args) != len(self.u_types_lst)):
                    raise ValueError("Insufficient arguments for \"u\" or \"U\" flag!")
                else:

                    self.sim_args = {
                        "data_type": self.data_type,
                        self.u_types_lst[0]: float(u_args[0]),
                        self.u_types_lst[1]: float(u_args[1])
                    }

                    if self.data_type == "static":
                        self.sim_args.update({self.u_types_lst[2]: float(u_args[2])})

                    self.data = self.data.get_viz_data_obj(self.sim_args)

        else:
            self.data = VisualizationData(args.FILE)

        # Load target fill ratios (inner parameter)
        try:
            args_tfr = args.TFR
        except Exception as e:
            try:
                args_tfr = args.tfr
            except Exception as e:
                args_tfr = None

        if args_tfr:
            if isinstance(args_tfr, list):
                if len(args_tfr) > 1: self.tfr = [float(i) for i in args_tfr] # multiple values in a list
                else: self.tfr = float(*args_tfr) # single value in a list
            else:
                self.tfr = float(args_tfr) # single scalar value
        else:
            self.tfr = self.data.tfr_range if isinstance(self.data, VisualizationData) else self.data.get_tfr_range()

        # Load sensor probabilities (inner parameter)
        try:
            args_sp = args.SP
        except Exception as e:
            try:
                args_sp = args.sp
            except Exception as e:
                args_sp = None

        if args_sp:
            if isinstance(args_sp, list):
                if len(args_sp) > 1: self.sp = [float(i) for i in args_sp] #multiple values in a list
                else: self.sp = float(*args_sp) # single value in a list
            else:
                self.sp = float(args_sp) # single scalar value
        else:
            self.sp = self.data.sp_range if isinstance(self.data, VisualizationData) else self.data.get_sp_range()

        # Truncate data based on steps
        try:
            args_steps = args.steps
        except Exception as e:
            args_steps = None

        if args_steps:
            self.truncate_data_steps(args_steps)

    def generate_plot(self, args):

        if args.viz_type == "series":

            # Check whether only one specific target fill ratio and sensor probability has been specified
            if isinstance(self.tfr, list) or isinstance(self.sp, list):
                raise ValueError("Only a single target fill ratio and a single sensor probability allowed!")

            else:
                if args.i: plot_individual_timeseries(self.tfr, self.sp, self.data, args.CONV)
                else: plot_timeseries(self.tfr, self.sp, self.data, args.a, args.CONV)

        elif args.viz_type == "heatmap":

            # Check whether to plot single or gridded heatmap
            if isinstance(self.data, VisualizationData): plot_heatmap_vd(self.data, args.CONV)
            else:
                plot_heatmap_vdg(
                    self.data,
                    self.u_types_lst[0],
                    [float(i) for i in args.row] if len(args.row) > 1 else float(*args.row),
                    self.u_types_lst[-1],
                    [float(i) for i in args.col] if len(args.col) > 1 else float(*args.col),
                    args.rstr,
                    args.cstr,
                    args.CONV
                )

        elif args.viz_type == "scatter":

            if not isinstance(self.data, VisualizationData): raise NotImplementedError("Scatter plot for VisualizationDataGroup not implemented!")
            else:
                self.sim_args.update({"tfr": self.tfr, "sp": self.sp, "ymax": args.ymax})

                plot_scatter(self.data, args.CONV, self.sim_args, args.i)

        elif args.viz_type == "decision":

            if not isinstance(self.data, VisualizationData): raise NotImplementedError("Collective decision plot for VisualizationDataGroup not implemented!")
            else:
                # Calculate the number of steps to evaluate the decisions at
                decision_sim_steps = np.arange(args.step_inc, self.data.num_steps+1, args.step_inc)

                self.sim_args.update({"tfr": self.tfr, "sp": self.sp, "bins": args.bins, "sim_steps": decision_sim_steps})

                plot_decision(self.data, self.sim_args)

    def truncate_data_steps(self, steps):
        """Truncate the number of simulation steps used in visualization of the data.

        Args:
            steps: number of steps to truncate the data to.
        """

        def truncate_vd_steps(data_obj: VisualizationData):
            """Internal function to process step truncation for the VisualizationData type object.
            """

            data_obj.num_steps = steps

            for tfr in data_obj.tfr_range:
                for sp in data_obj.sp_range:

                    # Update the SimStats object
                    stats = data_obj.stats_obj_dict[tfr][sp]

                    stats.x_hat_sample_mean = np.delete(stats.x_hat_sample_mean, removal_ind, axis=-1)
                    stats.x_bar_sample_mean = np.delete(stats.x_bar_sample_mean, removal_ind, axis=-1)
                    stats.x_sample_mean = np.delete(stats.x_sample_mean, removal_ind, axis=-1)

                    stats.alpha_sample_mean = np.delete(stats.alpha_sample_mean, removal_ind, axis=-1)
                    stats.rho_sample_mean = np.delete(stats.rho_sample_mean, removal_ind, axis=-1)
                    stats.gamma_sample_mean = np.delete(stats.gamma_sample_mean, removal_ind, axis=-1)

                    stats.x_hat_sample_std = np.delete(stats.x_hat_sample_std, removal_ind, axis=-1)
                    stats.x_bar_sample_std = np.delete(stats.x_bar_sample_std, removal_ind, axis=-1)
                    stats.x_sample_std = np.delete(stats.x_sample_std, removal_ind, axis=-1)

                    stats.alpha_sample_std = np.delete(stats.alpha_sample_std, removal_ind, axis=-1)
                    stats.rho_sample_std = np.delete(stats.rho_sample_std, removal_ind, axis=-1)
                    stats.gamma_sample_std = np.delete(stats.gamma_sample_std, removal_ind, axis=-1)

                    stats.x = np.delete(stats.x, removal_ind, axis=-1)
                    stats.gamma = np.delete(stats.gamma, removal_ind, axis=-1)

                    # Ensure that the dimensions are as expected
                    assert stats.x_hat_sample_mean.shape[-1] == steps + 1
                    assert stats.x_bar_sample_mean.shape[-1] == steps + 1
                    assert stats.x_sample_mean.shape[-1] == steps + 1

                    assert stats.alpha_sample_mean.shape[-1] == steps + 1
                    assert stats.rho_sample_mean.shape[-1] == steps + 1
                    assert stats.gamma_sample_mean.shape[-1] == steps + 1

                    assert stats.x_hat_sample_std.shape[-1] == steps + 1
                    assert stats.x_bar_sample_std.shape[-1] == steps + 1
                    assert stats.x_sample_std.shape[-1] == steps + 1

                    assert stats.alpha_sample_std.shape[-1] == steps + 1
                    assert stats.rho_sample_std.shape[-1] == steps + 1
                    assert stats.gamma_sample_std.shape[-1] == steps + 1

                    assert stats.x.shape[-1] == steps + 1
                    assert stats.gamma.shape[-1] == steps + 1

                    data_obj.stats_obj_dict[tfr][sp] = stats

                    # Update AggregateStats object (if it exists)
                    try:
                        agg_stats_obj = data_obj.agg_stats_dict[tfr][sp]

                        agg_stats_obj.x_hat_sample_mean = np.delete(agg_stats_obj.x_hat_sample_mean, removal_ind, axis=-1)
                        agg_stats_obj.x_bar_sample_mean = np.delete(agg_stats_obj.x_bar_sample_mean, removal_ind, axis=-1)
                        agg_stats_obj.x_sample_mean = np.delete(agg_stats_obj.x_sample_mean, removal_ind, axis=-1)

                        agg_stats_obj.alpha_sample_mean = np.delete(agg_stats_obj.alpha_sample_mean, removal_ind, axis=-1)
                        agg_stats_obj.rho_sample_mean = np.delete(agg_stats_obj.rho_sample_mean, removal_ind, axis=-1)
                        agg_stats_obj.gamma_sample_mean = np.delete(agg_stats_obj.gamma_sample_mean, removal_ind, axis=-1)

                        agg_stats_obj.x_hat_sample_std = np.delete(agg_stats_obj.x_hat_sample_std, removal_ind, axis=-1)
                        agg_stats_obj.x_bar_sample_std = np.delete(agg_stats_obj.x_bar_sample_std, removal_ind, axis=-1)
                        agg_stats_obj.x_sample_std = np.delete(agg_stats_obj.x_sample_std, removal_ind, axis=-1)

                        agg_stats_obj.alpha_sample_std = np.delete(agg_stats_obj.alpha_sample_std, removal_ind, axis=-1)
                        agg_stats_obj.rho_sample_std = np.delete(agg_stats_obj.rho_sample_std, removal_ind, axis=-1)
                        agg_stats_obj.gamma_sample_std = np.delete(agg_stats_obj.gamma_sample_std, removal_ind, axis=-1)

                        # Ensure that the dimensions are as expected
                        assert agg_stats_obj.x_hat_sample_mean.shape[-1] == steps + 1
                        assert agg_stats_obj.x_bar_sample_mean.shape[-1] == steps + 1
                        assert agg_stats_obj.x_sample_mean.shape[-1] == steps + 1

                        assert agg_stats_obj.alpha_sample_mean.shape[-1] == steps + 1
                        assert agg_stats_obj.rho_sample_mean.shape[-1] == steps + 1
                        assert agg_stats_obj.gamma_sample_mean.shape[-1] == steps + 1

                        assert agg_stats_obj.x_hat_sample_std.shape[-1] == steps + 1
                        assert agg_stats_obj.x_bar_sample_std.shape[-1] == steps + 1
                        assert agg_stats_obj.x_sample_std.shape[-1] == steps + 1

                        assert agg_stats_obj.alpha_sample_std.shape[-1] == steps + 1
                        assert agg_stats_obj.rho_sample_std.shape[-1] == steps + 1
                        assert agg_stats_obj.gamma_sample_std.shape[-1] == steps + 1

                        data_obj.agg_stats_dict[tfr][sp] = agg_stats_obj

                    except Exception as e:
                        pass

            return data_obj

        assert steps < self.data.num_steps

        removal_ind = range(steps+1, self.data.num_steps+1) # this takes the initial step into account

        if isinstance(self.data, VisualizationData): # VisualizationData object
            self.data = truncate_vd_steps(self.data)

        else: # VisualizationDataGroup object
            # Iterate until a VisualizationData object is reached
            for k1, v1 in self.data.viz_data_obj_dict.items(): # entering first level

                for k2, v2 in v1.items(): # entering second level

                    # Check to see if there's a third level (only applicable for static topologies)
                    if isinstance(v2, dict):
                        for k3, v3 in v2.items():
                            self.data.viz_data_obj_dict[k1][k2][k3] = truncate_vd_steps(
                                self.data.viz_data_obj_dict[k1][k2][k3]
                            )

                    else:
                        self.data.viz_data_obj_dict[k1][k2] = truncate_vd_steps(
                            self.data.viz_data_obj_dict[k1][k2]
                        )
