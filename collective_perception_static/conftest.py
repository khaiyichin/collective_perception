import pytest
from collective_perception_py import sim_modules as sm
from collective_perception_py import viz_modules as vm

"""
"""

# Default (arbitrary) values
TFR_RANGE = []
SP_RANGE = []
NUM_STEPS = 10000
NUM_TRIALS = 3
NUM_AGENTS = 10,
COMMS_GRAPH_STR = "full"
COMMS_PERIOD = 5
COMMS_PROB = 1.0
TFR = 0.85
SP = 0.95

class MASParametrizer:

    class DummySimParam:
        def __init__(self,
                     tfr_range,
                     sp_range,
                     num_steps,
                     num_trials,
                     num_agents,
                     comms_graph_str,
                     comms_period,
                     comms_prob):

            self.tfr_range = tfr_range
            self.sp_range = sp_range
            self.num_steps = num_steps
            self.num_trials = num_trials
            self.num_agents = num_agents
            self.comms_graph_str = comms_graph_str
            self.comms_period = comms_period
            self.comms_prob = comms_prob
            self.filename_suffix_1 = ""
            self.filename_suffix_2 = ""

    def __init__(
        self,
        sparam_tfr_range = [],
        sparam_sp_range = [],
        sparam_num_steps = 10000,
        sparam_num_trials = 3,
        sparam_num_agents = 10,
        sparam_comms_graph_str = "full",
        sparam_comms_period = 5,
        sparam_comms_prob = COMMS_PROB,
        mas_tfr = 0.75,
        mas_sp = 0.95
    ):

        self.sim_param = self.DummySimParam(
            sparam_tfr_range,
            sparam_sp_range,
            sparam_num_steps,
            sparam_num_trials,
            sparam_num_agents,
            sparam_comms_graph_str,
            sparam_comms_period,
            sparam_comms_prob
        )
        self.mas_tfr = mas_tfr
        self.mas_sp = mas_sp

        # self.sp_range = [ int( str(UNIFORM_DIST_SP_ENUM) + lower_bound + upper_bound ) ] # [distribution id, lower bound incl., upper bound excl.]
        # self.sp_range = [ int(str(NORMAL_DIST_SP_ENUM) + mean + var) ] # [distribution id, mean, variance]

# @pytest.fixture(scope="function")
# def custom_mas_comms_graph(comms_graph_str):

#     mas_parametrizer = MASParametrizer(
#         sparam_tfr_range = [],
#         sparam_sp_range = [],
#         sparam_num_steps = 10000,
#         sparam_num_trials = 3,
#         sparam_num_agents = 10,
#         sparam_comms_graph_str = comms_graph_str,
#         sparam_comms_period = 5,
#         sparam_comms_prob = COMMS_PROB,
#         mas_tfr = 0.75,
#         mas_sp = 0.95
#     )

#     return sm.MultiAgentSim(mas_parametrizer.sim_param, mas_parametrizer.mas_tfr, mas_parametrizer.mas_sp)

@pytest.fixture(scope="function")
def mas(request):
    return sm.MultiAgentSim(request.param.sim_param, request.param.mas_tfr, request.param.mas_sp)

@pytest.fixture(scope="function")
def mas_full(request):
    request.param.sim_param.comms_graph_str = "full"
    return sm.MultiAgentSim(request.param.sim_param, request.param.mas_tfr, request.param.mas_sp)

@pytest.fixture(scope="function")
def mas_ring(request):
    request.param.sim_param.comms_graph_str = "ring"
    return sm.MultiAgentSim(request.param.sim_param, request.param.mas_tfr, request.param.mas_sp)

@pytest.fixture(scope="function")
def mas_line(request):
    request.param.sim_param.comms_graph_str = "line"
    return sm.MultiAgentSim(request.param.sim_param, request.param.mas_tfr, request.param.mas_sp)

@pytest.fixture(scope="function")
def mas_sf(request):
    request.param.sim_param.comms_graph_str = "scale-free"
    return sm.MultiAgentSim(request.param.sim_param, request.param.mas_tfr, request.param.mas_sp)