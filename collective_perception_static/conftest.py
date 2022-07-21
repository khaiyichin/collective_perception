import pytest
import numpy as np

from collective_perception_py import sim_modules as sm
from collective_perception_py import viz_modules as vm

"""
"""

# Default (arbitrary) values
TFR_RANGE = []
SP_RANGE = []
NUM_STEPS = 10000
NUM_TRIALS = 3
NUM_AGENTS = 10
COMMS_GRAPH_STR = "full"
COMMS_PERIOD = 5
COMMS_PROB = 1.0
LEGACY=False
TFR = np.round( (0.99 - 0.01) * np.random.random_sample() + 0.01, 3 )
SP = np.round( (0.99 - 0.51) * np.random.random_sample() + 0.51, 3 )
TEST_REPEATS = 20

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
                     comms_prob,
                     legacy):

            self.tfr_range = tfr_range
            self.sp_range = sp_range
            self.num_steps = num_steps
            self.num_trials = num_trials
            self.num_agents = num_agents
            self.comms_graph_str = comms_graph_str
            self.comms_period = comms_period
            self.comms_prob = comms_prob
            self.legacy = legacy
            self.filename_suffix_1 = ""
            self.filename_suffix_2 = ""

    def __init__(
        self,
        sparam_tfr_range = [],
        sparam_sp_range = [],
        sparam_num_steps = NUM_STEPS,
        sparam_num_trials = NUM_TRIALS,
        sparam_num_agents = NUM_AGENTS,
        sparam_comms_graph_str = COMMS_GRAPH_STR,
        sparam_comms_period = COMMS_PERIOD,
        sparam_comms_prob = COMMS_PROB,
        sparam_legacy=LEGACY,
        mas_tfr = (0.99 - 0.51) * np.random.random_sample() + 0.51,
        mas_sp = (0.99 - 0.51) * np.random.random_sample() + 0.51
    ):

        self.sim_param = self.DummySimParam(
            sparam_tfr_range,
            sparam_sp_range,
            sparam_num_steps,
            sparam_num_trials,
            sparam_num_agents,
            sparam_comms_graph_str,
            sparam_comms_period,
            sparam_comms_prob,
            sparam_legacy
        )
        self.mas_tfr = mas_tfr
        self.mas_sp = mas_sp

        # self.sp_range = [ int( str(UNIFORM_DIST_SP_ENUM) + lower_bound + upper_bound ) ] # [distribution id, lower bound incl., upper bound excl.]
        # self.sp_range = [ int(str(NORMAL_DIST_SP_ENUM) + mean + var) ] # [distribution id, mean, variance]

@pytest.fixture(scope="function")
def mas(request):
    return sm.MultiAgentSim(request.param.sim_param, request.param.mas_tfr, request.param.mas_sp)

def assert_initial_agent_values(agent_obj):
        assert agent_obj.total_b_tiles_obs == 0 and isinstance(agent_obj.total_b_tiles_obs, int)
        assert agent_obj.total_obs == 0 and isinstance(agent_obj.total_obs, int)
        assert agent_obj.get_x_hat() == 0.5 # the initial x_hat is exactly 0.5
        assert agent_obj.get_alpha() == 0.0 # the initial alpha is exactly 0.0
        assert agent_obj.get_x_bar() == 0.0 # the initial value is 0.0
        assert agent_obj.get_rho() == 0.0 # the initial value is 0.0
        assert agent_obj.get_x() == 0.0 # the initial value is 0.0
        assert agent_obj.get_gamma() == 0.0 # the initial value is 0.0