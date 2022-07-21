import pytest
import conftest as cftest
from conftest import mas
import numpy as np
import scipy.stats as spy_stats
from collective_perception_py import sim_modules as sm

tfr_range_1 = np.linspace(0.55, 0.75, 3).tolist()
sp_range_1 = np.linspace(0.6, 0.8, 3).tolist()
num_steps_1 = 50000
num_trials_1 = 10
num_agents_1 = 15
comms_graph_str_1 = "ring"
comms_period_1 = 2
tfr_10 = tfr_range_1[0]
sp_10 = sp_range_1[0]

@pytest.mark.parametrize(
    "mas",
    [ cftest.MASParametrizer(
        sparam_tfr_range = tfr_range_1,
        sparam_sp_range = sp_range_1,
        sparam_num_steps = num_steps_1,
        sparam_num_trials = num_trials_1,
        sparam_num_agents = num_agents_1,
        sparam_comms_graph_str = comms_graph_str_1,
        sparam_comms_period = comms_period_1,
        sparam_comms_prob = cftest.COMMS_PROB,
        mas_tfr = tfr_10,
        mas_sp = sp_10
    ) ], indirect=True
)
def test_mas_initialization(mas): # @todo: add more test cases (use different parametrization)

    assert mas.sim_data.num_agents == num_agents_1
    assert mas.sim_data.num_trials == num_trials_1
    assert mas.sim_data.num_steps == num_steps_1 
    assert mas.sim_data.comms_period == comms_period_1
    
    assert mas.num_trials == num_trials_1

    sim_stats_local_mean_attributes = [
        mas.stats.x_hat_sample_mean,
        mas.stats.alpha_sample_mean,
        mas.stats.x_hat_sample_std,
        mas.stats.alpha_sample_std
    ]
    
    for attr in sim_stats_local_mean_attributes:
        assert len(attr) == num_trials_1
        assert len(attr[0]) == num_steps_1 + 1

    sim_stats_mean_social_attributes = [
        mas.stats.x_bar_sample_mean,
        mas.stats.rho_sample_mean,
        mas.stats.x_bar_sample_std,
        mas.stats.rho_sample_std
    ]

    for attr in sim_stats_mean_social_attributes:
        assert len(attr) == num_trials_1
        assert len(attr[0]) == num_steps_1//comms_period_1 + 1

    sim_stats_mean_informed_attributes = [
        mas.stats.x_sample_mean,
        mas.stats.gamma_sample_mean,
        mas.stats.x_sample_std,
        mas.stats.gamma_sample_std
    ]

    for attr in sim_stats_mean_informed_attributes:
        assert len(attr) == num_trials_1
        assert len(attr[0]) == num_steps_1//comms_period_1 + 1

    assert len(mas.stats.x) == num_trials_1
    assert len(mas.stats.x[0]) == num_agents_1
    assert len(mas.stats.x[0][0]) == num_steps_1//comms_period_1 + 1
    assert len(mas.stats.gamma) == num_trials_1
    assert len(mas.stats.gamma[0]) == num_agents_1
    assert len(mas.stats.gamma[0][0]) == num_steps_1//comms_period_1 + 1

    # Check actual agent number
    assert mas.sim_data.comms_network.graph.num_vertices() == num_agents_1

    # Check that all agents have the defined sensor probabilities
    for v in mas.sim_data.comms_network.graph.vertices():
        agent = mas.sim_data.comms_network.agents_vp[v]
        assert agent.b_prob == sp_10
        assert agent.w_prob == sp_10

    # Check that tile generation is correct
    tiles = mas.generate_tiles(num_agents_1) # num_agents x num_steps
    for agt_tiles in tiles:
        
        # Compare generated fill ratio with target fill ratio
        assert len(agt_tiles) == num_steps_1

        fill_ratio = sum(agt_tiles)/len(agt_tiles)
        np.testing.assert_approx_equal(fill_ratio, tfr_10, 2)

"""
Test full network generation
"""
@pytest.mark.parametrize(
    "mas", [cftest.MASParametrizer(sparam_comms_graph_str="full")], indirect=True
)
def test_full_comms_network(mas):
    n = mas.sim_data.num_agents
    graph = mas.sim_data.comms_network.graph

    # Check the graph properties
    assert graph.num_edges() == n*(n-1)/2 # number of edges
    assert np.all(graph.get_total_degrees(graph.get_vertices()) == n-1) # number of neighbors

"""
Test ring network generation
"""
@pytest.mark.parametrize(
    "mas", [cftest.MASParametrizer(sparam_comms_graph_str="ring")], indirect=True
)
def test_ring_network(mas):
    n = mas.sim_data.num_agents
    graph = mas.sim_data.comms_network.graph

    # Check the graph properties
    assert graph.num_edges() == n # number of edges
    assert np.all(graph.get_total_degrees(graph.get_vertices()) == 2) # number of neighbors

    # Verify neighbors
    for v in graph.get_vertices():
        neighbors = sorted(graph.get_all_neighbors(v))

        if v == 0:
            assert neighbors[0] == v+1
            assert neighbors[1] == n-1
        elif v == n-1:
            assert neighbors[0] == 0
            assert neighbors[1] == v-1
        else:
            assert neighbors[0] == v-1
            assert neighbors[1] == v+1

"""
Test line network generation
"""
@pytest.mark.parametrize(
    "mas", [cftest.MASParametrizer(sparam_comms_graph_str="line")], indirect=True
)
def test_line_network(mas):
    n = mas.sim_data.num_agents
    graph = mas.sim_data.comms_network.graph
    degrees_lst = graph.get_total_degrees(graph.get_vertices())

    # Check the graph properties
    assert graph.num_edges() == n-1 # number of edges
    assert degrees_lst[0] == 1 # first agent
    degrees_lst = np.delete(degrees_lst, 0)
    assert degrees_lst[-1] == 1 # last agent
    degrees_lst = np.delete(degrees_lst, -1)
    assert np.all(degrees_lst == 2) # number of neighbors for remaining agents

    # Verify neighbors
    for v in graph.get_vertices():
        neighbors = sorted(graph.get_all_neighbors(v))

        if v == 0:
            assert neighbors[0] == v+1
            assert len(neighbors) == 1
        elif v == n-1:
            assert neighbors[0] == v-1
            assert len(neighbors) == 1
        else:
            assert neighbors[0] == v-1
            assert neighbors[1] == v+1

# @TODO: don't have a good way to verify the scale-free network yet
# """
# Test scale-free network generation
# """
# @pytest.mark.parametrize(
#     "mas", [cftest.MASParametrizer(
#         sparam_comms_graph_str="scale-free",
#         sparam_num_agents=100)], indirect=True
# )
# def test_sf_network(mas):
#     n = mas.sim_data.num_agents
#     graph = mas.sim_data.comms_network.graph
#     degrees_lst = graph.get_total_degrees(graph.get_vertices())

#     import graph_tool.draw as gt_draw

#     gt_draw.graph_draw(graph)

#     deg_set = sorted(set(degrees_lst)) # get unique degree values
#     num_occur_lst = [ len([i for i in degrees_lst if i == deg]) for deg in deg_set ]
#     for deg in deg_set:

#         # Gather the number of occurrence for a particular degree
#         num_occur = len([i for i in degrees_lst if i == deg])

#         # Compute the rate of occurrence
#         rate_occur = num_occur / n

#         # Compute the theoretical probability (https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model#Degree_distribution)
#         prob = 1/(deg ** 3)

#         # Compare rate with the theoretical probability --> this isn't the correct way to verify
#         # np.testing.assert_approx_equal(rate_occur, prob, 3)

#     pass


# Check that the uniformly distributed sensor probability is 0.75

# test agent class?

"""
Test local value computation
"""
@pytest.mark.parametrize("mas", [cftest.MASParametrizer(sparam_num_steps=20)], indirect=True)
def test_local_vals(mas):

    # Execute simulation fully
    mas.run()

    # Create alias variables
    b = mas.sim_data.b_prob
    w = mas.sim_data.w_prob

    for _ in range(cftest.TEST_REPEATS):

        # Obtain random indices to test
        trial_ind = np.random.randint(0, mas.sim_data.num_trials)
        agent_ind = np.random.randint(0, mas.sim_data.num_agents)
        step_ind = np.random.randint(0, mas.sim_data.num_steps)

        # Get observations
        observations = mas.sim_data.agent_obs[trial_ind][agent_ind][:step_ind+1]

        # Compute local values manually
        total_obs = len(observations)
        total_b_obs = sum(observations)

        if total_b_obs <= (1-w)*total_obs:
            local_est = 0.0

            num_conf = np.square(b + w - 1) * (total_obs * np.square(w) - 2 * (total_obs - total_b_obs) * w + total_obs - total_b_obs )
            denom_conf = np.square(w) * np.square(w - 1)
            local_conf = num_conf / denom_conf

        elif total_b_obs >= b*total_obs:
            local_est = 1.0

            num_conf = np.square(b + w - 1) * (total_obs * np.square(b) - 2 * total_b_obs * b + total_b_obs )
            denom_conf = np.square(b) * np.square(b - 1)
            local_conf = num_conf / denom_conf

        else:
            num_est = total_b_obs/total_obs + w - 1.0
            denom_est = b + w - 1.0
            local_est = num_est/denom_est

            num_conf = np.power(total_obs, 3) * np.square(b + w - 1)
            denom_conf = total_b_obs * (total_obs - total_b_obs)
            local_conf = num_conf / denom_conf

        # Verify values
        np.testing.assert_allclose(local_est, mas.x_hat[trial_ind][agent_ind][step_ind+1]) # the first value is from the initial estimate, not from observations
        np.testing.assert_allclose(local_conf, mas.alpha[trial_ind][agent_ind][step_ind+1]) # the first value is from the initial estimate, not from observations

"""
Test social value computation
"""
@pytest.mark.parametrize(
    "mas",
    [cftest.MASParametrizer(sparam_num_steps=40, sparam_comms_graph_str="full"),
     cftest.MASParametrizer(sparam_num_steps=40, sparam_comms_graph_str="ring"),
     cftest.MASParametrizer(sparam_num_steps=40, sparam_comms_graph_str="line"),
     cftest.MASParametrizer(sparam_num_steps=40, sparam_comms_graph_str="scale-free")],
    indirect=True
)
def test_social_vals(mas):

    # Execute simulation fully
    mas.run()

    comms_network_obj = mas.sim_data.comms_network

    for _ in range(cftest.TEST_REPEATS):
        # Obtain random indices to test
        trial_ind = np.random.randint(0, mas.sim_data.num_trials)
        agent_ind = np.random.randint(0, mas.sim_data.num_agents)
        step_ind = np.random.randint(0, mas.sim_data.num_steps//mas.sim_data.comms_period + 1)

        # Get agent's neighbors' values
        neighbors = comms_network_obj.graph.get_all_neighbors(agent_ind)
        estimates = []
        confidences = []

        for neighbor in neighbors:
            estimates.append(mas.x_hat[trial_ind][neighbor][step_ind*mas.sim_data.comms_period])
            confidences.append(mas.alpha[trial_ind][neighbor][step_ind*mas.sim_data.comms_period])

        # Compute social values manually
        social_est = 0.0 if all([i == 0 for i in confidences]) else np.average(estimates, weights=confidences)
        social_conf = np.mean(confidences)

        np.testing.assert_allclose(mas.x_bar[trial_ind][agent_ind][step_ind], social_est)
        np.testing.assert_allclose(mas.rho[trial_ind][agent_ind][step_ind], social_conf)

"""
Test legacy social value computation
"""
@pytest.mark.parametrize(
    "mas",
    [cftest.MASParametrizer(sparam_num_steps=20, sparam_comms_graph_str="full", sparam_legacy=True),
     cftest.MASParametrizer(sparam_num_steps=20, sparam_comms_graph_str="ring", sparam_legacy=True),
     cftest.MASParametrizer(sparam_num_steps=20, sparam_comms_graph_str="line", sparam_legacy=True),
     cftest.MASParametrizer(sparam_num_steps=20, sparam_comms_graph_str="scale-free", sparam_legacy=True)],
    indirect=True
)
def test_social_vals_legacy(mas):

    # Execute simulation fully
    mas.run()

    comms_network_obj = mas.sim_data.comms_network

    for _ in range(cftest.TEST_REPEATS):
        # Obtain random indices to test
        trial_ind = np.random.randint(0, mas.sim_data.num_trials)
        agent_ind = np.random.randint(0, mas.sim_data.num_agents)
        step_ind = np.random.randint(0, mas.sim_data.num_steps//mas.sim_data.comms_period + 1)

        # Get agent's neighbors' values
        neighbors = comms_network_obj.graph.get_all_neighbors(agent_ind)
        estimates = []
        confidences = []

        for neighbor in neighbors:
            estimates.append(mas.x_hat[trial_ind][neighbor][step_ind*mas.sim_data.comms_period])
            confidences.append(mas.alpha[trial_ind][neighbor][step_ind*mas.sim_data.comms_period])

        # Compute social values manually
        social_est = np.mean(estimates)
        social_conf = spy_stats.hmean(confidences)

        np.testing.assert_allclose(mas.x_bar[trial_ind][agent_ind][step_ind], social_est)
        np.testing.assert_allclose(mas.rho[trial_ind][agent_ind][step_ind], social_conf)

"""
Test informed value computation
"""
@pytest.mark.parametrize(
    "mas",
    [cftest.MASParametrizer(sparam_num_steps=40, sparam_comms_graph_str="full"),
     cftest.MASParametrizer(sparam_num_steps=40, sparam_comms_graph_str="ring"),
     cftest.MASParametrizer(sparam_num_steps=40, sparam_comms_graph_str="line"),
     cftest.MASParametrizer(sparam_num_steps=40, sparam_comms_graph_str="scale-free")],
    indirect=True
)
def test_informed_vals(mas):

    # Execute simulation fully
    mas.run()

    for _ in range(cftest.TEST_REPEATS):
        # Obtain random indices to test
        trial_ind = np.random.randint(0, mas.sim_data.num_trials)
        agent_ind = np.random.randint(0, mas.sim_data.num_agents)
        step_ind = np.random.randint(0, mas.sim_data.num_steps//mas.sim_data.comms_period + 1)

        # Obtain local and social values
        local_est = mas.x_hat[trial_ind][agent_ind][step_ind*mas.sim_data.comms_period]
        local_conf = mas.alpha[trial_ind][agent_ind][step_ind*mas.sim_data.comms_period]
        social_est =  mas.x_bar[trial_ind][agent_ind][step_ind]
        social_conf =  mas.rho[trial_ind][agent_ind][step_ind]

        # Compute informed values manually
        informed_est = (local_conf * local_est + social_est) / (local_conf + social_conf)
        informed_conf = np.mean([local_conf, social_conf])

        print("debug", step_ind, social_conf, local_conf)

        np.testing.assert_allclose(mas.x[trial_ind][agent_ind][step_ind], informed_est)
        np.testing.assert_allclose(mas.gamma[trial_ind][agent_ind][step_ind], informed_conf)

"""
Test legacy informed value computation
"""
@pytest.mark.parametrize(
    "mas",
    [cftest.MASParametrizer(sparam_num_steps=20, sparam_comms_graph_str="full", sparam_legacy=True),
     cftest.MASParametrizer(sparam_num_steps=20, sparam_comms_graph_str="ring", sparam_legacy=True),
     cftest.MASParametrizer(sparam_num_steps=20, sparam_comms_graph_str="line", sparam_legacy=True),
     cftest.MASParametrizer(sparam_num_steps=20, sparam_comms_graph_str="scale-free", sparam_legacy=True)],
    indirect=True
)
def test_informed_vals_legacy(mas):

    # Execute simulation fully
    mas.run()

    for _ in range(cftest.TEST_REPEATS):
        # Obtain random indices to test
        trial_ind = np.random.randint(0, mas.sim_data.num_trials)
        agent_ind = np.random.randint(0, mas.sim_data.num_agents)
        step_ind = np.random.randint(0, mas.sim_data.num_steps//mas.sim_data.comms_period + 1)

        # Obtain local and social values
        local_est = mas.x_hat[trial_ind][agent_ind][step_ind*mas.sim_data.comms_period]
        local_conf = mas.alpha[trial_ind][agent_ind][step_ind*mas.sim_data.comms_period]
        social_est =  mas.x_bar[trial_ind][agent_ind][step_ind]
        social_conf =  mas.rho[trial_ind][agent_ind][step_ind]

        # Compute informed values manually
        informed_est = (local_conf * local_est + social_conf * social_est) / (local_conf + social_conf)
        informed_conf = local_conf + social_conf

        np.testing.assert_allclose(mas.x[trial_ind][agent_ind][step_ind], informed_est)
        np.testing.assert_allclose(mas.gamma[trial_ind][agent_ind][step_ind], informed_conf)

# test that the uniform distribution is actually uniform

"""
Test agent sensor accuracy
"""
@pytest.mark.parametrize(
    "mas",
    [cftest.MASParametrizer(sparam_num_trials=2, sparam_num_agents=3, sparam_comms_period=50)],
    indirect=True
)
def test_agent_sensor_prob(mas):

    # Execute simulation fully
    mas.run()

    for _ in range(cftest.TEST_REPEATS):

        # Obtain random indices to test
        trial_ind = np.random.randint(0, mas.sim_data.num_trials)
        agent_ind = np.random.randint(0, mas.sim_data.num_agents)

        # Extract the tiles and observations
        tiles = mas.sim_data.tiles[trial_ind][agent_ind]
        obs = mas.sim_data.agent_obs[trial_ind][agent_ind]

        assert len(tiles) == mas.sim_data.num_steps
        assert len(obs) == mas.sim_data.num_steps

        # Gather all the matches between the tiles and observations
        correct_obs = sum([1 for t, o in zip(tiles, obs) if t == o])
        obs_rate = correct_obs/len(obs)

        np.testing.assert_approx_equal(obs_rate, mas.sim_data.b_prob, 2)
        np.testing.assert_approx_equal(obs_rate, mas.sim_data.w_prob, 2)

"""
Test agent reset function
"""
@pytest.mark.parametrize(
    "mas", [ cftest.MASParametrizer() ], indirect=True)
def test_reset_agents(mas):

    comms_network_obj = mas.sim_data.comms_network
    agents_vertices = comms_network_obj.graph.vertices() # agent IDs

    # Verify initial values
    [cftest.assert_initial_agent_values(comms_network_obj.agents_vp[a]) for a in agents_vertices]

    # Run the simulation for one trial
    mas.run_sim(0)

    # Verify post simulation and agent reset
    for a in agents_vertices:
        agent_obj = comms_network_obj.agents_vp[a]

        # Verify that simulation has been executed
        assert (agent_obj.total_b_tiles_obs <= mas.sim_data.num_steps and agent_obj.total_b_tiles_obs >= 0)
        assert agent_obj.total_obs == mas.sim_data.num_steps
        assert abs(agent_obj.get_x_hat() - 0.5) > 0.0 # the initial x_hat is exactly 0.5
        assert agent_obj.get_alpha() > 0.0 # the initial alpha is exactly 0.0
        assert agent_obj.get_x_bar() > 0.0 # the initial value is 0.0
        assert agent_obj.get_rho() > 0.0 # the initial value is 0.0
        assert agent_obj.get_x() > 0.0 # the initial value is 0.0
        assert agent_obj.get_gamma() > 0.0 # the initial value is 0.0
        if agent_obj.total_b_tiles_obs > 1: assert agent_obj.get_avg_black_obs() > 0.0 # the initial value is 0.0

        # Reset agent and verify
        agent_obj.reset()
        cftest.assert_initial_agent_values(agent_obj)