#!/usr/bin/env python3
import collective_perception_py.viz_modules as vm
import collective_perception_py.viz_modules_benchmark as vmb
import argparse
import pickle


def load_decd_data(filepath):
    # Load and assert that data has the following form:
    with open(filepath, "rb") as fopen:
        obj = pickle.load(fopen)

    return obj



def main():
    parser = argparse.ArgumentParser(
        description=
        "Load .decd data to provide performance comparison between benchmark algorithms and the minimalistic collective perception algorithm."
    )

    parser.add_argument(
        "REF",
        type=str,
        help="path to the reference .decd data (exported from minimalistic collective perception .vdg data)"
    )
    parser.add_argument(
        "BM",
        type=str,
        nargs="+",
        help="the target fill ratio for the experiment that produced the data to be extracted"
    )
    parser.add_argument(
        "--sp",
        type=float,
        nargs="+",
        help="space-delimited array of sensor probabilities to plot (default: all sensor probabilities included in the .decd files)"
    )
    parser.add_argument(
        "--leg",
        type=str,
        nargs="+",
        help="space-delimited array of legends that matches the description of the input benchmark .decd files; an additional value can be provided at the beginning if a legend title is desired (defaul: file names)"
    )
    parser.add_argument(
        "-s",
        type=str,
        help="suffix for the output figure filename"
    )
    parser.add_argument(
        "--ymax",
        type=float,
        help="y-axis limit for plot"
    )

    args = parser.parse_args()

    # Extract reference data
    ref_dec_data_dict = load_decd_data(args.REF)

    # Extract benchmark data
    bm_dec_data_dicts = {}
    ref_sim_steps = ref_dec_data_dict["meta_data"]["extracted_sim_steps"]
    max_sim_step = max(ref_sim_steps)

    for bm_path in args.BM:
        bm_dec_data_dicts[bm_path] = load_decd_data(bm_path)

        # Get the max sim step for the benchmark data
        bm_sim_steps = bm_dec_data_dicts[bm_path]["meta_data"]["extracted_sim_steps"]
        max_sim_step = min(max(bm_sim_steps), max_sim_step)

        # Check to see if the steps match up after controlling for the max_sim_step
        matches = [a == b for a, b in 
                   zip(ref_sim_steps, bm_sim_steps) if a <= max_sim_step and b <= max_sim_step]
        if not all(matches):
            raise Exception("The simulation steps of the decision data do not match up.")

    # Get simulation steps to plot
    sim_steps = [i for i in ref_sim_steps if i <= max_sim_step]
    sim_steps.pop(0) # no meaning for us to know the performance on the 0th time step

    # Filter sensor probabilities if needed
    if args.sp:
        sp_range = args.sp
    else:
        sp_range = ref_dec_data_dict["meta_data"]["sp"]

    # Check legend input
    if args.leg:
        assert(len(args.leg) == len(args.BM) or len(args.leg) == len(args.BM) + 1)

    args_plot_perf_ratio = {
        "benchmark_str": list(bm_dec_data_dicts.items())[0][1]["meta_data"]["data_type"],
        "tfr": ref_dec_data_dict["meta_data"]["tfr"],
        "sensor_probability_range": sp_range,
        "num_trials": ref_dec_data_dict["meta_data"]["num_trials"],
        "sim_steps": sim_steps,
        "speed": ref_dec_data_dict["meta_data"]["speed"],
        "density": ref_dec_data_dict["meta_data"]["density"],
        "num_options": ref_dec_data_dict["meta_data"]["num_options"],
        "colorbar_label": "Sensor Accuracies",
        "legends": None if not args.leg else args.leg,
        "ymax": None if not args.ymax else args.ymax,
        "output_suffix": "" if not args.s else "_{0}".format(args.s)
    }

    vmb.plot_decision_performance_ratio(ref_dec_data_dict,
                                        bm_dec_data_dicts,
                                        args_plot_perf_ratio)
    
if __name__ == "__main__": main()