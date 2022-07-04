#include "simulation_stats_set.hpp"

SimulationStatsSet::SimulationStatsSet(collective_perception_cpp::proto::SimulationStatsSet &sim_stats_set_msg)
{
    // Load simulation parameters
    SimulationSet::Deserialize(sim_stats_set_msg);

    // Load simulation statistics
    for (auto itr = sim_stats_set_msg.mutable_stats_packets()->begin();
         itr != sim_stats_set_msg.mutable_stats_packets()->end();
         ++itr)
    {
        StatsPacket packet;
        PacketMsg *packet_msg_ptr = itr->mutable_packet();

        packet.sim_type = packet_msg_ptr->sim_type();
        packet.comms_range = packet_msg_ptr->comms_range();
        packet.target_fill_ratio = packet_msg_ptr->tfr();
        packet.b_prob = packet_msg_ptr->b_prob();
        packet.w_prob = packet_msg_ptr->w_prob();
        packet.num_agents = packet_msg_ptr->num_agents();
        packet.num_trials = packet_msg_ptr->num_trials();
        packet.num_steps = packet_msg_ptr->num_steps();
        packet.density = packet_msg_ptr->density();
        packet.speed = packet_msg_ptr->speed();

        // Load Stats messages (local, social, and informed)
        RTStatsProtoMsg *rts_msg_ptr = itr->mutable_rts();
        auto &local_vals_msg = rts_msg_ptr->local_vals();
        auto &social_vals_msg = rts_msg_ptr->social_vals();
        auto &informed_vals_msg = rts_msg_ptr->informed_vals();
        auto &sp_mean_vals_msg = rts_msg_ptr->sp_mean_vals();

        // Iterate through different trials of values
        for (size_t i = 0; i < num_trials_; ++i)
        {
            Stats local, social, informed;

            // Copy local values
            std::copy(local_vals_msg.Get(i).x_mean().begin(),
                      local_vals_msg.Get(i).x_mean().end(),
                      std::back_insert_iterator(local.x_sample_mean));
            std::copy(local_vals_msg.Get(i).conf_mean().begin(),
                      local_vals_msg.Get(i).conf_mean().end(),
                      std::back_insert_iterator(local.confidence_sample_mean));
            std::copy(local_vals_msg.Get(i).x_std().begin(),
                      local_vals_msg.Get(i).x_std().end(),
                      std::back_insert_iterator(local.x_sample_std));
            std::copy(local_vals_msg.Get(i).conf_std().begin(),
                      local_vals_msg.Get(i).conf_std().end(),
                      std::back_insert_iterator(local.confidence_sample_std));

            packet.repeated_local_values.push_back(local);

            // Copy social values
            std::copy(social_vals_msg.Get(i).x_mean().begin(),
                      social_vals_msg.Get(i).x_mean().end(),
                      std::back_insert_iterator(social.x_sample_mean));
            std::copy(social_vals_msg.Get(i).conf_mean().begin(),
                      social_vals_msg.Get(i).conf_mean().end(),
                      std::back_insert_iterator(social.confidence_sample_mean));
            std::copy(social_vals_msg.Get(i).x_std().begin(),
                      social_vals_msg.Get(i).x_std().end(),
                      std::back_insert_iterator(social.x_sample_std));
            std::copy(social_vals_msg.Get(i).conf_std().begin(),
                      social_vals_msg.Get(i).conf_std().end(),
                      std::back_insert_iterator(social.confidence_sample_std));

            packet.repeated_social_values.push_back(social);

            // Copy informed values
            std::copy(informed_vals_msg.Get(i).x_mean().begin(),
                      informed_vals_msg.Get(i).x_mean().end(),
                      std::back_insert_iterator(informed.x_sample_mean));
            std::copy(informed_vals_msg.Get(i).conf_mean().begin(),
                      informed_vals_msg.Get(i).conf_mean().end(),
                      std::back_insert_iterator(informed.confidence_sample_mean));
            std::copy(informed_vals_msg.Get(i).x_std().begin(),
                      informed_vals_msg.Get(i).x_std().end(),
                      std::back_insert_iterator(informed.x_sample_std));
            std::copy(informed_vals_msg.Get(i).conf_std().begin(),
                      informed_vals_msg.Get(i).conf_std().end(),
                      std::back_insert_iterator(informed.confidence_sample_std));

            packet.repeated_informed_values.push_back(informed);

            // Copy random sensor probability mean values
            std::copy(sp_mean_vals_msg.begin(),
                      sp_mean_vals_msg.end(),
                      std::back_insert_iterator(packet.sp_mean_values));
        }

        // Insert populated StatsPacket object
        InsertStatsPacket(packet);
    }
}

void SimulationStatsSet::Serialize(collective_perception_cpp::proto::SimulationStatsSet &sim_stats_set_msg)
{
    // Serialize SimulationSet object
    SimulationSet::Serialize(sim_stats_set_msg);

    // Serialize StatsPacket objects for varying simulation parameters
    std::vector<StatsPacket> stats_packet_vec = SimulationSet::VectorizePackets(packets_);

    for (auto stats_packet_itr = stats_packet_vec.begin(); stats_packet_itr != stats_packet_vec.end(); ++stats_packet_itr)
    {
        auto stats_packet_msg_ptr = sim_stats_set_msg.add_stats_packets();
        PacketMsg *packet_msg_ptr = stats_packet_msg_ptr->mutable_packet();

        packet_msg_ptr->set_sim_type(simulation_type_);
        packet_msg_ptr->set_tfr(stats_packet_itr->target_fill_ratio);
        packet_msg_ptr->set_b_prob(stats_packet_itr->b_prob);
        packet_msg_ptr->set_w_prob(stats_packet_itr->w_prob);

        packet_msg_ptr->set_comms_range(comms_range_);
        packet_msg_ptr->set_num_agents(num_agents_);
        packet_msg_ptr->set_num_trials(num_trials_);
        packet_msg_ptr->set_num_steps(num_steps_);
        packet_msg_ptr->set_density(density_);
        packet_msg_ptr->set_speed(speed_);

        *((stats_packet_msg_ptr->mutable_rts())->mutable_sp_mean_vals()) =
            {stats_packet_itr->sp_mean_values.begin(), stats_packet_itr->sp_mean_values.end()};

        // Store RepeatedTrialStats object for repeated trials
        *(stats_packet_msg_ptr->mutable_rts()) = ExtractRepeatedTrialStatsMsg({stats_packet_itr->repeated_local_values,
                                                                               stats_packet_itr->repeated_social_values,
                                                                               stats_packet_itr->repeated_informed_values});
    }
}

RTStatsProtoMsg SimulationStatsSet::ExtractRepeatedTrialStatsMsg(const std::array<RepeatedTrials<Stats>, 3> &arr)
{
    RTStatsProtoMsg rts_msg;
    std::vector<StatsProtoMsg> local_stats_msg_vec(num_trials_);
    std::vector<StatsProtoMsg> social_stats_msg_vec(num_trials_);
    std::vector<StatsProtoMsg> informed_stats_msg_vec(num_trials_);

    // Iterate through each trial
    for (size_t ind = 0; ind < num_trials_; ++ind)
    {
        // arr[0][j] is the Stats struct of the jth trial in the repeated_local_values vector (i.e., jth element in the vector)
        // arr[1][j] is the Stats struct of the jth trial in the repeated_social_values vector (i.e., jth element in the vector)
        // arr[2][j] is the Stats struct of the jth trial in the repeated_informed_values vector (i.e., jth element in the vector)

        StatsProtoMsg local, social, informed;

        *(local.mutable_x_mean()) = {arr[0][ind].x_sample_mean.begin(), arr[0][ind].x_sample_mean.end()};
        *(social.mutable_x_mean()) = {arr[1][ind].x_sample_mean.begin(), arr[1][ind].x_sample_mean.end()};
        *(informed.mutable_x_mean()) = {arr[2][ind].x_sample_mean.begin(), arr[2][ind].x_sample_mean.end()};

        *(local.mutable_conf_mean()) = {arr[0][ind].confidence_sample_mean.begin(), arr[0][ind].confidence_sample_mean.end()};
        *(social.mutable_conf_mean()) = {arr[1][ind].confidence_sample_mean.begin(), arr[1][ind].confidence_sample_mean.end()};
        *(informed.mutable_conf_mean()) = {arr[2][ind].confidence_sample_mean.begin(), arr[2][ind].confidence_sample_mean.end()};

        *(local.mutable_x_std()) = {arr[0][ind].x_sample_std.begin(), arr[0][ind].x_sample_std.end()};
        *(social.mutable_x_std()) = {arr[1][ind].x_sample_std.begin(), arr[1][ind].x_sample_std.end()};
        *(informed.mutable_x_std()) = {arr[2][ind].x_sample_std.begin(), arr[2][ind].x_sample_std.end()};

        *(local.mutable_conf_std()) = {arr[0][ind].confidence_sample_std.begin(), arr[0][ind].confidence_sample_std.end()};
        *(social.mutable_conf_std()) = {arr[1][ind].confidence_sample_std.begin(), arr[1][ind].confidence_sample_std.end()};
        *(informed.mutable_conf_std()) = {arr[2][ind].confidence_sample_std.begin(), arr[2][ind].confidence_sample_std.end()};

        local_stats_msg_vec[ind] = local;
        social_stats_msg_vec[ind] = social;
        informed_stats_msg_vec[ind] = informed;
    }

    *(rts_msg.mutable_local_vals()) = {local_stats_msg_vec.begin(), local_stats_msg_vec.end()};
    *(rts_msg.mutable_social_vals()) = {social_stats_msg_vec.begin(), social_stats_msg_vec.end()};
    *(rts_msg.mutable_informed_vals()) = {informed_stats_msg_vec.begin(), informed_stats_msg_vec.end()};

    return rts_msg;
}