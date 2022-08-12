#!/usr/bin/env python3
import numpy as np
import os
from joblib import Parallel, delayed

import collective_perception_py.sim_modules as sm
import collective_perception_py.viz_modules as vm
import argparse

def create_vdg_obj(obj_filepath: str):

    vd = vm.VisualizationData()

    # Load the object
    ext = os.path.splitext(obj_filepath)[1]

    if ext == ".pkl":
        obj = vd.load_pkl_file(obj_filepath, True)
        vdg = vm.VisualizationDataGroupStatic()

    elif ext == ".pbs":
        obj = vd.load_proto_file(obj_filepath)
        vdg = vm.VisualizationDataGroupDynamic()

    else: raise RuntimeError("Unknown extension encountered; please provide \".pkl\" or \".pbs\" files.")

    # Populate VisualizationData object parameters
    vd.sp_range = obj.sp_range
    vd.tfr_range = obj.tfr_range

    vd.num_agents = obj.num_agents
    vd.num_trials = obj.num_trials
    vd.num_steps = obj.num_steps
    vd.stats_obj_dict = {}
    vd.sim_data_obj_dict = obj.sim_data_obj_dict # adding a temporary attribute to instance

    if obj.sim_type == "dynamic":
        vd.comms_range = -1.0
        vd.density = obj.density
        vd.speed = obj.speed
        vd.comms_period = 1
        vdg.viz_data_obj_dict[vd.speed] = {vd.density: vd}

    elif obj.sim_type == "static":
        vd.graph_type = obj.graph_type
        vd.comms_period = 1
        vd.comms_prob = -1
        vdg.viz_data_obj_dict[vd.comms_period] = {vd.comms_prob: {vd.num_agents: vd}}

    else: raise RuntimeError("Unknown simulation type for loaded object!")

    return vdg

def parallel_inner_loop_trial(sp, num_agents, num_steps, agent_obs_trial, sim_cls_obj):
    h_arr = np.cumsum(agent_obs_trial, axis=1) # total black tiles observed by all agents in a single trial; shape of (num_agents, num_steps)

    est = np.zeros((num_agents, num_steps+1))
    conf = np.zeros((num_agents, num_steps+1))

    # Iterate through each agent
    for a in range(num_agents):

        # Store initial local values
        est[a][0] = 0.5
        conf[a][0] = 0.0

        # Compute local estimates and confidences based on the extracted tiles and observations
        est[a][1:] = np.asarray( [ sim_cls_obj.compute_x_hat(h, t, sp, sp) for h, t in zip(h_arr[a], range(1, num_steps+1)) ] )
        conf[a][1:] = np.asarray( [ sim_cls_obj.compute_fisher_hat(h, t, sp, sp) for h, t in zip(h_arr[a], range(1, num_steps+1)) ] )

    return (est, conf)

def main():
    # Arguments to collect ExperimentData pickle file
    parser = argparse.ArgumentParser(description="Extract local estimates from a single ExperimentData or SimulationStatsSet file into a VisualizationDataGroupStatic file.")

    parser.add_argument("FILE", type=str, help="path to the pickled ExperimentData or SimulationStatsSet file")
    parser.add_argument("-s", type=str, help="path to store the pickled VisualizationDataGroup object")

    args = parser.parse_args()

    # Load experiment data to extract tiles and observations
    vdg = create_vdg_obj(args.FILE)
    
    # Extract the only VisualizationData object in the VisualizationDataGroup
    vd = list(list(vdg.viz_data_obj_dict.values())[0].values())[0]

    # Try to extract once more since the static class has an additional dictionary
    try:
        vd = list(vd.values())[0]

    except Exception as e:
        pass

    dummy_sim_cls_obj = sm.Sim()

    # Iterate through each target fill ratio
    for tfr in vd.tfr_range:

        # Initialize the stats_obj_dict for the VisualizationData instance
        vd.stats_obj_dict = {tfr: {}}

        for sp in vd.sp_range:

            # Create data container to store computed local stats
            stats = sm.Sim.SimStats(
                "multi",
                vd.num_trials,
                vd.num_steps,
                vd.comms_period,
                vd.num_agents,
                False
            )

            # Extract agent observations
            agent_obs = vd.sim_data_obj_dict[tfr][sp].agent_obs

            assert agent_obs.shape[0] == vd.num_trials
            assert agent_obs.shape[1] == vd.num_agents
            assert agent_obs.shape[2] == vd.num_steps

            # Run computation for each trial in parallel
            outputs_lst = Parallel(n_jobs=-1, verbose=10)(
                delayed(parallel_inner_loop_trial)(sp, vd.num_agents, vd.num_steps, agent_obs[t], dummy_sim_cls_obj) for t in range(vd.num_trials)
            ) # outputs list of estimates and confidences for each trial

            stats.x = np.asarray([i for i, _ in outputs_lst])
            stats.gamma = np.asarray([i for _, i in outputs_lst])

            del vd.sim_data_obj_dict[tfr][sp]

            vd.stats_obj_dict[tfr].update({sp: stats})

    del vd.sim_data_obj_dict

    # Modify the output filename to specify that output VisualizationDataGroup contains local
    if args.s: args.s += "_LOCAL"
    else: args.s = "vdg_LOCAL.pkl"

    vdg.stored_obj_counter += 1 # update counter

    vdg.save(args.s)

if __name__ == "__main__":
    main()