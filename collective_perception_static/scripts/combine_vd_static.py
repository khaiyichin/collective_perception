#!/usr/bin/env python3

# Script to combine two VDs into one along either the 'tfr_range' parameter or the 'sp_range' parameter.
# Their viz_data_obj_dict (i.e., their stored VisualizationData objects) MUST share the same:
#   - comm_network type (enforced externally by user)
#   - number of trials
#   - comms probability
#   - number of agent range
#   - comms_period range
# and one of the following:
#   - tfr_range, or sp_range.
# 
# For example, 2 VDs that have varying target fill ratio ranges can only be combined if their:
#   - number of trials
#   - comms probability
#   - number of agent range
#   - comms_period range
#   - sp_range
#   are exactly the same.
# 
# Similarly, 2 VDs that have varying sensor probabilities can only be combined if their:
#   - number of trials
#   - comms probability
#   - number of agent range
#   - comms_period range
#   - tfr_range
# are exactly the same.

import collective_perception_py.viz_modules as vm
import argparse

def combine_vd(base: vm.VisualizationData,
               part: vm.VisualizationData,
               args,
               along_sp=True):

    # Merge at the VisualizationData level
    output_vdg = vm.VisualizationDataGroupStatic()

    if along_sp: # add new sp data

        # for all the base's visualization data's tfr range, add the new sp data
        base.sp_range.extend(part.sp_range)
        [base.stats_obj_dict[i].update(part.stats_obj_dict[i]) for i in part.tfr_range]
        base.sp_range = sorted(base.sp_range)

    else: # add new tfr data

        # for all the base's visualization data's sp range, add the new tfr data
        base.tfr_range.extend(part.tfr_range)
        base.stats_obj_dict.update(part.stats_obj_dict)
        base.tfr_range = sorted(base.tfr_range)

    output_vdg.viz_data_obj_dict[args.PERIOD] = { 
        args.COMMS_PROB: {
            args.NUM_AGENTS: base
        }
    }
    output_vdg.stored_obj_counter += 1

    return output_vdg

def main():
    parser = argparse.ArgumentParser(description="Load and combine two VisualizationData pickled objects to produce a VisualizationDataGroup object.\
        \nThis script can also be used to convert a single a VisualizationData into a VisualizationDataGroup.")
    parser.add_argument("BASE", type=str, help="path to the base VisualationDataGroupStatic pickled object")
    parser.add_argument("PERIOD", type=int, help="specific communications period")
    parser.add_argument("COMMS_PROB", type=float, help="specific communications probability")
    parser.add_argument("NUM_AGENTS", type=int, help="specific number of agents")
    parser.add_argument("--part", type=str, help="path to the other VisualationDataGroupStatic pickled object to be added to the base")
    parser.add_argument("-s", type=str, help="path to store the pickled VisualizationData object (without extension)")
    args = parser.parse_args()

    # Load base data
    base_vd = vm.VisualizationData.load(args.BASE)

    # Check to see if combining or just pure conversion of base file
    if args.part:

        # Load the part data
        part_vd = vm.VisualizationData.load(args.part)

        if any([a != b for a, b in zip(base_vd.sp_range, part_vd.sp_range)]): # sp_range doesn't match

            assert all([a == b for a,b in zip(base_vd.tfr_range, part_vd.tfr_range)])

            output_vdg = combine_vd(base_vd, part_vd, args, True)

        elif any([a != b for a, b in zip(base_vd.tfr_range, part_vd.tfr_range)]): # tfr_range doesn't match

            assert all([a == b for a,b in zip(base_vd.sp_range, part_vd.sp_range)])

            output_vdg = combine_vd(base_vd, part_vd, args, False)

        else:
            raise RuntimeError("Both the target fill ratio and sensor probability ranges do not match!")

    else: # pure conversion
        output_vdg = vm.VisualizationDataGroupStatic()
        output_vdg.viz_data_obj_dict[args.PERIOD] = {
            args.COMMS_PROB: {
                args.NUM_AGENTS: base_vd
            }
        }
        output_vdg.stored_obj_counter += 1

    output_vdg.save(args.s)

if __name__ == "__main__":
    main()