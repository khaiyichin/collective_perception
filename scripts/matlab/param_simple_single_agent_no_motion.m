%% Parameter file for simple_single_agent_no_motion.m

N = 1000;                       % total number of observations
b = 0.45;                        % sensor probability to black tile
w = b;                          % sensor probability to white tile
sim_cycles = 200;                % number of agents to simulate (or
                                % simulation cycles for one agent)
desired_fill_ratio = 0.95;   % fill ratio, f (can be set to rand(1))