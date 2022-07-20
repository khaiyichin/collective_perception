import pytest
import conftest as cftest
from conftest import mas
import numpy as np
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

# @TODO: don't have a good way to verify the scale-free network yet
# """
# Test scale-free network generation
# """
# @pytest.mark.parametrize(
#     "mas", [cftest.MASParametrizer(
#         sparam_comms_graph_str="scale-free",
#         sparam_num_agents=10000)], indirect=True
# )
# def test_sf_network(mas):
#     n = mas.sim_data.num_agents
#     graph = mas.sim_data.comms_network.graph
#     degrees_lst = graph.get_total_degrees(graph.get_vertices())

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

# test that the comms network is correct (neighbor numbers, number of agents etc)

# test that the correct comms network is used: ring is ring, full is full etc.

# Check that the uniformly distributed sensor probability is 0.75

# test agent class?

# test local value computation (using only a single agent)

# test social value computation

# test informed value computation

# test that the uniform distribution is actually uniform

# test that the agents are actually resetted

# ensure that agent values are num_steps + 1