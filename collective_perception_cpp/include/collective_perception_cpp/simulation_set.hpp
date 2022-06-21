#ifndef SIMULATION_SET_HPP
#define SIMULATION_SET_HPP

#include "data_common.hpp"

template <typename T>
using PacketDict = std::unordered_map<unsigned int, std::unordered_map<unsigned int, T>>;

class SimulationSet
{
public:
    inline void PopulateSimulationSetParams(const SimulationSet &s)
    {
        simulation_type_ = s.simulation_type_;
        num_agents_ = s.num_agents_;
        num_trials_ = s.num_trials_;
        num_steps_ = s.num_steps_;
        comms_range_ = s.comms_range_;
        density_ = s.density_;
        speed_ = s.speed_;
        tfr_range_ = s.tfr_range_;
        sp_range_ = s.sp_range_;
    }

    std::string simulation_type_ = "dynamic";

    unsigned int num_agents_;

    unsigned int num_trials_;

    unsigned int num_steps_;

    float comms_range_;

    float density_;

    float speed_;

    std::vector<float> tfr_range_;

    std::vector<float> sp_range_;

    std::string proto_filepath_;

protected:
    template <typename T>
    inline void Deserialize(T &msg)
    {
        auto sim_set_msg_ptr = msg.mutable_sim_set();

        simulation_type_ = sim_set_msg_ptr->sim_type(); // simuation type
        num_agents_ = sim_set_msg_ptr->num_agents();    // number of agents
        num_trials_ = sim_set_msg_ptr->num_trials();    // number of experiments
        num_steps_ = sim_set_msg_ptr->num_steps();      // number of simulation steps
        comms_range_ = sim_set_msg_ptr->comms_range();  // communication range
        density_ = sim_set_msg_ptr->density();          // swarm density
        speed_ = sim_set_msg_ptr->speed();              // speed

        std::copy(sim_set_msg_ptr->tfr_range().begin(),
                  sim_set_msg_ptr->tfr_range().end(),
                  std::back_inserter(tfr_range_)); // range of target fill ratios
        std::copy(sim_set_msg_ptr->sp_range().begin(),
                  sim_set_msg_ptr->sp_range().end(),
                  std::back_inserter(sp_range_)); // range of sensor probabilities
    }

    template <typename T>
    inline void Serialize(T &msg)
    {
        auto sim_set_msg_ptr = msg.mutable_sim_set();

        sim_set_msg_ptr->set_sim_type(simulation_type_); // simulation type
        sim_set_msg_ptr->set_num_agents(num_agents_);    // number of agents
        sim_set_msg_ptr->set_num_trials(num_trials_);    // number of experiments
        sim_set_msg_ptr->set_num_steps(num_steps_);      // number of simulation steps
        sim_set_msg_ptr->set_comms_range(comms_range_);  // communication range
        sim_set_msg_ptr->set_density(density_);          // swarm density
        sim_set_msg_ptr->set_speed(speed_);              // swarm speed

        *(sim_set_msg_ptr->mutable_tfr_range()) = {tfr_range_.begin(), tfr_range_.end()};
        *(sim_set_msg_ptr->mutable_sp_range()) = {sp_range_.begin(), sp_range_.end()};
    }

    template <typename T>
    inline T GetPacket(const float &tfr, const float &sp, PacketDict<T> &packets_ref)
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

        return packets_ref[tfr_internal][sp_internal];
    }

    template <typename T>
    inline void InsertPacket(const T &packet, PacketDict<T> &packets_ref)
    {
        int tfr_internal = RoundToMultiple(ConvertToInternalUnits(packet.target_fill_ratio), 5);
        int sp_internal = RoundToMultiple(ConvertToInternalUnits(packet.b_prob), 5);

        packets_ref[tfr_internal][sp_internal] = packet;
    }

    /**
     * @brief Convert the provided value into internal units: 0.005 = 5
     *
     * @param val The value in physical units to be converted
     * @return int The converted value in internal units
     */
    inline int ConvertToInternalUnits(const float &val) { return std::round(val * 1e3); }

    /**
     * @brief Round an integer to the next multiple of a specific base
     *
     * Examples:
     *      roundToMultiple(13, 10) = 10
     *      roundToMultiple(14, 4) = 16
     *
     * @param value The value to be rounded
     * @param base Rounding base such that the result is a multiple of the base
     * @return int The rounded value
     */
    inline int RoundToMultiple(const int &value, const int &base)
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

    template <typename T>
    inline std::vector<T> VectorizePackets(PacketDict<T> &packets_ref)
    {
        std::vector<T> packets;

        for (auto &tfr_key_value : packets_ref)
        {
            for (auto &sp_key_value : tfr_key_value.second)
            {
                packets.push_back(sp_key_value.second);
            }
        }

        return packets;
    }
};

#endif