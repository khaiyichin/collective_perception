#include "simulation_data_set.hpp"

SimulationDataSet::SimulationDataSet(collective_perception_cpp::proto::SimulationDataSet &exp_data_proto)
{
}

void SimulationDataSet::Serialize(collective_perception_cpp::proto::SimulationDataSet &exp_data_proto)
{
    exp_data_proto.set_num_agents();
    exp_data_proto.set_num_exp();
}

SimPacket SimulationDataSet::GetSimPacket(const float &tfr, const float &sp)
{
    /* Internal units description

    The values for tfr and sp range from 0.000 to 1.000, with a resolution of 0.005.
    Therefore, after scaling to integers, the resolution for the internal units would be 5, and the new
    range is between 0 to 1000.
    */

    // Convert to internal units to use as keys (thus avoiding trailing values with floats)
    unsigned int tfr_internal, sp_internal;

    tfr_internal = RoundToMultiple(ConvertToInternalUnits(tfr), 5);
    sp_internal = RoundToMultiple(ConvertToInternalUnits(sp), 5);

    return sim_packets[tfr_internal][sp_internal];
}

void SimulationDataSet::InsertSimPacket(const SimPacket &packet)
{
    int tfr_internal = RoundToMultiple(ConvertToInternalUnits(packet.target_fill_ratio), 5);
    int sp_internal = RoundToMultiple(ConvertToInternalUnits(packet.b_prob), 5);

    sim_packets[tfr_internal][sp_internal] = packet;
}

int SimulationDataSet::RoundToMultiple(const int &value, const int &base)
{
    int absBase = std::abs(base);
    int remainder = std::abs(value) % absBase;

    if (2 * remainder >= absBase)
    {
        return std::round(std::copysign(std::abs(value) + absBase - remainder, value));
    }
    else
    {
        return std::copysign(std::abs(value) - remainder, value);
    }
}