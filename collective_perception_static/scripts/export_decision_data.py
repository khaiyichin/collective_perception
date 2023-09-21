#!/usr/bin/env python3
import collective_perception_py.viz_modules as vm
import collective_perception_py.viz_modules_benchmark as vmb
import argparse
import os
import json
import pprint

def load_data(args):
    data_obj = []
    decision_data = {}
    sim_steps = 0
    vdg_obj = None

    # Check if data to load is a VDG file (minimalistic collective perception data) or a folder of JSON files (benchmark data)
    if os.path.isdir(args.FILE): # JSON data

        # Pick one random json file to read the `sim_type` field
        sim_type = ""
        found = False

        for root, _, files in os.walk(args.FILE):
            for f in files:

                if os.path.splitext(f)[1] == ".json":
                    with open(os.path.join(root, f), "r") as file:
                        json_dict = json.load(file)
                        sim_type = json_dict["sim_type"]
                    found = True
                    break

            if found: break

        args.FOLDER = args.FILE # add the .FOLDER field so that the Visualizer object can use it
        
        # Convert JSON files into the Visualizer object so that we can extract the decision data
        if sim_type == vmb.Valentini2016Visualizer.BENCHMARK_STR:
            args.SP = args.PARAM_2_RANGE # add the .sp field so that the Visualizer object can use it
            data_obj = vmb.Valentini2016Visualizer(args)

        elif sim_type == vmb.Crosscombe2017Visualizer.BENCHMARK_STR:
            args.FRR = args.PARAM_2_RANGE # add the .FRR field so that the Visualizer object can use it
            data_obj = vmb.Crosscombe2017Visualizer(args)

        elif sim_type == vmb.Ebert2020Visualizer.BENCHMARK_STR:
            args.SP = args.PARAM_2_RANGE # add the .sp field so that the Visualizer object can use it
            data_obj = vmb.Ebert2020Visualizer(args)

        else:
            raise Exception("Invalid benchmark algorithm files!")

        # Extract only the decision data based on the sim steps desired
        if data_obj.num_steps < args.step_inc: raise Exception("Step increments is too large for number of timesteps available.")
        sim_steps = [i for i in range(0, data_obj.num_steps + 1, args.step_inc)]
        decision_data = data_obj.get_decision_data(sim_steps)

    else: # VDG data
        if not args.u: raise Exception("The \"-u\" arguments must be provided for .vdg files.")

        try:
            vdg_obj = vm.VisualizationDataGroupStatic.load(args.FILE)
            vdg_args = {"data_type": "static", "comms_period": args.u[0], "comms_prob": args.u[1], "num_agents": args.u[2]}
            
        except Exception as e:
            try:
                vdg_obj = vm.VisualizationDataGroupDynamic.load(args.FILE)
                vdg_args = {"data_type": "dynamic", "speed": args.u[0], "density": args.u[1]}

            except Exception as e:
                print("Invalid .vdg file!")
                print(e)
                exit(-1)

        try:
            data_obj = vdg_obj.get_viz_data_obj(vdg_args)
        except KeyError as e:
            print("VDG dictionary key not found:", e, ";please check that the provided arguments are correct.")
            exit(-1)

        try:
            # Extract only the decision data based on the sim steps desired
            if data_obj.num_steps < args.step_inc: raise Exception("Step increments is too large for number of timesteps available.")
            sim_steps = [i for i in range(0, data_obj.num_steps + 1, args.step_inc)]
            decision_data = {sp: {} for sp in args.PARAM_2_RANGE}

            # Iterate through each sim step to obtain decision fraction
            for ss in sim_steps:
                sp_dec_frac_dict = data_obj.get_decision_fractions(args.TFR, args.PARAM_2_RANGE, ss, args.nopt)
                for sp, dec_frac in sp_dec_frac_dict.items():
                    decision_data[sp][ss] = dec_frac

        except KeyError as e:
            print("Invalid key! KeyError:", e)
            print("Available target fill ratios are:", data_obj.tfr_range)
            print("Available sensor probabilities are:", data_obj.sp_range)
            exit(-1)

    # Extract meta data
    meta_data = {
        "tfr": args.TFR,
        "num_agents": data_obj.num_agents,
        "num_steps": data_obj.num_steps,
        "num_trials": data_obj.num_trials,
        "num_options": args.nopt,
        "extracted_sim_steps": sim_steps
    }

    if vdg_obj:
        meta_data["data_type"] = vdg_args["data_type"]
        meta_data["sp"] = args.PARAM_2_RANGE

        if isinstance(vdg_obj, vm.VisualizationDataGroupStatic):
            meta_data["comms_period"] = vdg_args["comms_period"]
            meta_data["comms_prob"] = vdg_args["comms_prob"]
        else:
            meta_data["speed"] = vdg_args["speed"]
            meta_data["density"] = vdg_args["density"]

    else:
        meta_data["data_type"] = data_obj.BENCHMARK_STR
        meta_data[data_obj.BENCHMARK_PARAM_ABBR] = args.PARAM_2_RANGE
        meta_data["speed"] = data_obj.speed
        meta_data["density"] = data_obj.density

    return {"meta_data": meta_data, "dec_data": decision_data}

def main():
    parser = argparse.ArgumentParser(
        description="Export decision data from a VisualizationDataGroup object (minimalistic collective perception data) or JSON files from a single benchmark algorithm (benchmark data).\
            \nThe exported .decd file contains a dict of dicts in the following form:\
            \n{\
            \n    \'meta_data\':\
            \n    {\
            \n        'data_type': (str),\
            \n        'density': (float),\
            \n        'extracted_sim_steps': (list of ints),\
            \n        'num_agents': (int),\
            \n        'num_options': (int),\
            \n        'num_steps': (int),\
            \n        'num_trials': (int),\
            \n        'sp': (list of floats),\
            \n        'speed': (float),\
            \n        'tfr': (float)\
            \n    },\
            \n    'dec_data':\
            \n    {\
            \n        sensor_prob_1 (float):\
            \n        {\
            \n            sim_step_11 (int): decision_11 (float),\
            \n            sim_step_12 (int): decision_12 (float),\
            \n            ...\
            \n        },\
            \n        sensor_prob_2 (float):\
            \n        {\
            \n            sim_step_21 (int): decision_21 (float),\
            \n            sim_step_22 (int): decision_22 (float),\
            \n            ...\
            \n        },\
            \n        sensor_prob_3 (float):\
            \n        {\
            \n            sim_step_31 (int): decision_31 (float),\
            \n            sim_step_32 (int): decision_32 (float),\
            \n            ...\
            \n        },\
            \n        ...\
            \n    }\
            \n}",
    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "FILE",
        type=str,
        help="path to the .vdg file (minimalistic collective perception data) or a folder containing the .json files (benchmark data)"
    )
    parser.add_argument(
        "TFR",
        type=float,
        help="the target fill ratio for the experiment that produced the data to be extracted"
    )
    parser.add_argument(
        "PARAM_2_RANGE",
        type=float,
        nargs="+",
        help="space-delimited array of values for the 2nd parameter to extract (typically sensor probabilities)"
    )
    parser.add_argument(
        "--step_inc",
        type=int,
        default=1000,
        help="(optional) the increment in simulation steps to evaluate decisions (default: 1000)"
    )
    parser.add_argument(
        "--nopt",
        type=int,
        default=10,
        help="(optional) the number of options/bins to evaluate decisions (default: 10)"
    )
    parser.add_argument(
        "-s",
        type=str,
        help="path to store the pickle .decd object (without extension)"
    )
    parser.add_argument(
        "-u",
        type=float,
        nargs="+",
        help="(communications period, communication probability, number of agents)/(robot speed, swarm density) to use in extracting static/dynamic data"
    )

    args = parser.parse_args()

    # Load decd data object (actually just a combination of dictionaries)
    data = load_data(args)

    print("\nProcessed decision data:")
    pprint.pprint(data)

    vm.serialize_decision_data(data, args.s)

if __name__ == "__main__": main()