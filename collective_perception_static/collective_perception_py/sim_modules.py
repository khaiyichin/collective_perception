import yaml
import numpy as np
import scipy.stats as spy_stats
from datetime import datetime
import graph_tool as gt
import graph_tool.generation as gt_gen
import graph_tool.stats as gt_stats
import random
import pickle
import os
import warnings

BLACK_TILE = 1
WHITE_TILE = 0
POSINF = 1.0e+100
UNIFORM_DIST_SP_ENUM = -2
NORMAL_DIST_SP_ENUM = -3

warnings.filterwarnings("ignore", category=RuntimeWarning) # suppress warnings since division by zero occur frequently here

def decode_sp_distribution(encoded_val):

    assert encoded_val < 0.0

    # Decode distribution parameters
    dist_id = int( str(encoded_val)[:2] )
    param_1 = round( float( str( encoded_val )[2:6] ) * 1e-3, 3 )
    param_2 = round( float( str( encoded_val )[6:] ) * 1e-3, 3 )

    return dist_id, param_1, param_2

def encode_sp_distribution(dist_id, param_1, param_2):

    # Encode distribution parameters into single int in a list
    p1_enc = "{:04d}".format( int( round(param_1, 3) * 1e3 ) ) # 4 digits, scaled by 1e3
    p2_enc = "{:04d}".format( int( round(param_2, 3) * 1e3 ) ) # 4 digits, scaled by 1e3
    encoded = [ int( str(dist_id) + p1_enc + p2_enc ) ] # [ int(distribution id, parameter 1, parameter 2) ]

    return encoded

class Sim:
    """Top level simulation class.
    """

    class SimData:
        """Class for storing simulation data that can be used replicate experimental results.
        """

        def __init__(self, sim_type, num_trials, num_agents, num_steps, sensor_prob, comms_period):
            self.sim_type = sim_type
            self.num_trials = num_trials
            self.num_agents = num_agents
            self.num_steps = num_steps
            self.b_prob = sensor_prob # P(black|black)
            self.w_prob = sensor_prob # P(white|white)
            self.comms_network_str = None
            self.comms_period = comms_period
            self.comms_network = None

            if sim_type == "multi":
                self.tiles = np.zeros( (num_trials, num_agents, num_steps) )
                self.agent_obs = np.zeros( (num_trials, num_agents, num_steps) )

    class SimStats:
        """Class for storing statistics of simulation experiments.
        """

        def __init__(self, sim_type, num_trials=0, num_steps=0, comms_period=1, num_agents=0, legacy=False):

            # Initialize data containers so that it can be also used to populate dynamic simulation data @TODO: not a good way to do this, please revise!
            self.sp_distribution = None
            self.sp_distributed_sample_mean = None

            self.x_hat_sample_mean = None
            self.alpha_sample_mean = None
            self.x_hat_sample_std = None
            self.alpha_sample_std = None

            self.x_bar_sample_mean = None
            self.rho_sample_mean = None
            self.x_bar_sample_std = None
            self.rho_sample_std = None

            self.x_sample_mean = None
            self.gamma_sample_mean = None
            self.x_sample_std = None
            self.gamma_sample_std = None
            self.legacy = legacy

            if sim_type == "multi" and num_trials != 0 and num_steps != 0:
                self.sp_distribution = None
                self.sp_distributed_sample_mean = np.zeros(num_trials)

                # @todo: temporary hack to show individual robot values; in the future this should be stored
                # elsewhere
                self.x = np.zeros( (num_trials, num_agents, num_steps + 1) )
                self.gamma = np.zeros( (num_trials, num_agents, num_steps + 1) )

                self.x_hat_sample_mean = np.zeros( (num_trials, num_steps + 1) )
                self.alpha_sample_mean = np.zeros( (num_trials, num_steps + 1) )
                self.x_hat_sample_std = np.zeros( (num_trials, num_steps + 1) )
                self.alpha_sample_std = np.zeros( (num_trials, num_steps + 1) )

                self.x_bar_sample_mean = np.zeros( (num_trials, num_steps//comms_period + 1) )
                self.rho_sample_mean = np.zeros( (num_trials, num_steps//comms_period + 1) )
                self.x_bar_sample_std = np.zeros( (num_trials, num_steps//comms_period + 1) )
                self.rho_sample_std = np.zeros( (num_trials, num_steps//comms_period + 1) )

                self.x_sample_mean = np.zeros( (num_trials, num_steps + 1) )
                self.gamma_sample_mean = np.zeros( (num_trials, num_steps + 1) )
                self.x_sample_std = np.zeros( (num_trials, num_steps + 1) )
                self.gamma_sample_std = np.zeros( (num_trials, num_steps + 1) )

    def __init__(self, num_trials=0, num_steps=0, targ_fill_ratio=0.0, main_filename_suffix=""):
        self.num_trials = num_trials
        self.num_steps = num_steps
        self.targ_fill_ratio = targ_fill_ratio
        self.avg_fill_ratio = 0.0
        self.tiles_record = np.empty( (self.num_trials, self.num_steps) )
        self.main_filename_suffix = main_filename_suffix

    def generate_tiles(self, num_agents=1):
        """Generate the tiles based on the desired/nominal fill ratios.

        Args:
            num_agents: Number of tiles rows to generate, only used in multi-agent simulations.

        Returns:
            A 1-D (single agent simulation) or a (num_agents x self.num_steps) 2-D numpy array
             of binary tiles (multi-agent simulation).
        """

        # Draw bernoulli samples for tiles based on desired fill ratio
        if num_agents > 1:
            tiles = np.random.binomial(1, self.targ_fill_ratio * np.ones((num_agents, self.num_steps)))

        else:
            tiles = np.random.binomial(1, self.targ_fill_ratio * np.ones(self.num_steps) )

            assert(len(tiles) == self.num_steps)

        return tiles

    def observe_color(self, tile_color, color_prob):
        """Provide observation of a tile color for one agent.
        """

        if (np.random.uniform() < color_prob):
            return tile_color
        else:
            return 1.0 - tile_color

    def compute_x_hat(self, h, t, b, w):
        """Compute the fill ratio estimate for one agent.
        """

        if h <= (1.0 - w) * t:
            return 0.0
        elif h >= b*t:
            return 1.0
        else:
            return (h/t + w - 1.0) / (b + w - 1)

    def compute_fisher_hat(self, h, t, b, w):
        """Compute the Fisher information for one agent.
        """

        if (b == 1.0) and (w == 1.0): np.power(t, 3) / (h * (t - h)) # perfect sensor
        else: # imperfect sensor
            if h <= (1.0 - w) * t:
                num = np.square(b + w - 1.0) * (t * np.square(w) - 2 * (t - h) * w + (t - h))
                denom = np.square(w) * np.square(w - 1.0)
                output = num / denom
            elif h >= b*t:
                num = np.square(b + w - 1.0) * (t * np.square(b) - 2 * h * b + h)
                denom = np.square(b) * np.square(b - 1.0)
                output = num / denom
            else:
                num = np.power(t, 3) * np.square(b + w - 1.0)
                denom = h * (t - h)
                output = num / denom

        return np.nan_to_num(output, posinf=POSINF)

    def compute_x_bar(self, x_arr, weights, legacy=False): # TODO: need to split this out of the parent class since it should be modular (i.e., we may not use the same social function)

        # Nullify the social estimate if weights are all 0
        if all([i == 0 for i in weights]): return 0.0

        if legacy:
            return np.mean(x_arr)
        else:
            return np.average(x_arr, weights=weights)

    def compute_fisher_bar(self, fisher_arr, legacy=False):

        # Nullify the social confidence if weights are all 0
        if all([i == 0 for i in fisher_arr]): return 0.0

        if legacy: return np.nan_to_num( spy_stats.hmean(fisher_arr), posinf=POSINF )
        else: return np.nan_to_num( np.sum(fisher_arr), posinf=POSINF )

    def compute_x(self, x_hat, alpha, x_bar, rho): # TODO: need to split this out of the parent class since it should be modular (i.e., we may not use the same objective function)

        # Check if the confidences are zero
        if alpha == 0 and rho == 0: return x_hat # use the local estimate since that's the best possible guess at this time

        return ( alpha*x_hat + rho*x_bar ) / (alpha + rho)

    def compute_fisher(self, alpha, rho):
        return alpha + rho

class MultiAgentSim(Sim):

    def __init__(self, sim_param_obj, targ_fill_ratio, sensor_prob):

        num_agents = sim_param_obj.num_agents
        num_trials = sim_param_obj.num_trials
        num_steps = sim_param_obj.num_steps
        comms_period = sim_param_obj.comms_period
        legacy = sim_param_obj.legacy

        super().__init__(num_trials, num_steps, targ_fill_ratio, sim_param_obj.filename_suffix_1)

        # Initialize data containers (to be serialized)
        self.stats = self.SimStats("multi", num_trials, num_steps, comms_period, num_agents, legacy)
        if sensor_prob < 0: # not actually the sensor probability; actually encoded distribution

            # Decode distribution parameters
            dist_id, param_1, param_2 = decode_sp_distribution(sensor_prob)

            # Store parameters
            self.generator = np.random.default_rng()
            self.dist_params = [param_1, param_2]
            self.sim_data = self.SimData("multi", num_trials, num_agents, num_steps, dist_id, comms_period)
        else:
            self.sim_data = self.SimData("multi", num_trials, num_agents, num_steps, sensor_prob, comms_period)

        # Initialize non-persistent simulation data
        self.x_hat = np.zeros( (num_trials, num_agents, num_steps + 1) )
        self.alpha = np.zeros( (num_trials, num_agents, num_steps + 1) )
        self.x_bar = np.zeros( (num_trials, num_agents, num_steps//comms_period + 1) )
        self.rho = np.zeros( (num_trials, num_agents, num_steps//comms_period + 1) )
        self.x = np.zeros( (num_trials, num_agents, num_steps + 1) )
        self.gamma = np.zeros( (num_trials, num_agents, num_steps + 1) )

        # Setup up communication graph
        self.setup_comms_graph(sim_param_obj.comms_graph_str, sim_param_obj.comms_prob)
        self.create_agents(legacy)

    def setup_comms_graph(self, graph_type, comms_prob):

        # Add edges depending on the type of graph desired
        self.sim_data.comms_network_str = graph_type

        if graph_type == "full":
            self.sim_data.comms_network = self._create_complete_graph()
        elif graph_type == "line":
            self.sim_data.comms_network = self._create_line_graph()
        elif graph_type == "ring":
            self.sim_data.comms_network = self._create_ring_graph()
        elif graph_type == "scale-free":
            self.sim_data.comms_network = self._create_scale_free_graph()

        # Add a new property map for communication probabilities
        comms_prob_eprop = self.sim_data.comms_network.graph.new_edge_property("double") # returns an EdgePropertyMap object pointing to the comms_graph

        # TODO: allow for different comm probabilities
        comms_prob_eprop.get_array()[:] = comms_prob # assign a probability of 1.0 to each edge for now

        # Store the communication probabilities into the CommsNetwork object
        self.sim_data.comms_network.comms_prob_ep = comms_prob_eprop

    def _create_complete_graph(self):
        """Create a fully-connected graph.
        """
        return CommsNetwork(gt_gen.complete_graph(self.sim_data.num_agents, directed=False))

    def _create_ring_graph(self):
        """Create a ring graph.
        """
        return CommsNetwork(gt_gen.circular_graph(self.sim_data.num_agents, k=1, directed=False))

    def _create_line_graph(self):
        """Create a line graph.
        """
        g, _ = gt_gen.geometric_graph([[i] for i in range(self.sim_data.num_agents)], 1)
        return CommsNetwork(g)

    def _create_scale_free_graph(self):
        """Create a scale-free graph.

        The graph is generated based on the Barabási-Albert network model with gamma = 1.
        Therefore, the degree distribution has the following form: Prob ~ k^-3, where k
        is the degree of a node/vertex.
        """
        return CommsNetwork(gt_gen.price_network(self.sim_data.num_agents, directed=False))

    def create_agents(self, legacy=False):

        # Create agent objects
        agents_vprop = self.sim_data.comms_network.graph.new_vertex_property("object") # need to populate agents into the vertices
        sensor_probs = []

        for vertex in self.sim_data.comms_network.graph.get_vertices():
            if self.sim_data.b_prob == UNIFORM_DIST_SP_ENUM: # sensor probability is an encoded value for a uniform distributed sensor probability
                b_sensor_prob = ( (self.dist_params[1] - self.dist_params[0]) * self.generator.random(self.num_trials) + self.dist_params[0] ).tolist()
                w_sensor_prob = b_sensor_prob

                sensor_probs.append(b_sensor_prob)

                self.stats.sp_distribution = "uniform"

            elif self.sim_data.b_prob == NORMAL_DIST_SP_ENUM: # sensor probability is an encoded value for a normal distributed sensor probability
                b_sensor_prob = ( self.generator.normal(self.dist_params[0], np.sqrt(self.dist_params[1]), self.num_trials) ).tolist()
                w_sensor_prob = b_sensor_prob

                sensor_probs.append(b_sensor_prob)

                self.stats.sp_distribution = "normal"

            else: # homogeneous sensor probability
                b_sensor_prob = self.sim_data.b_prob
                w_sensor_prob = self.sim_data.w_prob
                sensor_probs = [b_sensor_prob]

            agents_vprop[vertex] = Agent(b_sensor_prob, w_sensor_prob,
                                        (self.compute_x_hat, self.compute_fisher_hat),
                                        (self.compute_x_bar, self.compute_fisher_bar),
                                        (self.compute_x, self.compute_fisher),
                                        legacy)

        # @todo hacky solution to replace/update mean sensor probability
        if self.stats.sp_distribution:
            self.sim_data.b_prob = sensor_probs
            self.sim_data.w_prob = sensor_probs

        self.stats.sp_distributed_sample_mean = np.mean(sensor_probs, axis=0)

        self.sim_data.comms_network.agents_vp = agents_vprop

    def compute_sample_mean(self, trial_index):
        """Compute the sample mean for the results per trial.

        Args:
            trial_index: Index of the trial.
        """

        self.stats.x_hat_sample_mean[trial_index] = np.mean(self.x_hat[trial_index], axis=0)
        self.stats.alpha_sample_mean[trial_index] = np.mean(self.alpha[trial_index], axis=0)

        self.stats.x_bar_sample_mean[trial_index] = np.mean(self.x_bar[trial_index], axis=0)
        self.stats.rho_sample_mean[trial_index] = np.mean(self.rho[trial_index], axis=0)

        self.stats.x_sample_mean[trial_index] = np.mean(self.x[trial_index], axis=0)
        self.stats.gamma_sample_mean[trial_index] = np.mean(self.gamma[trial_index], axis=0)

    def compute_sample_std(self, trial_index):
        """Compute the sample standard deviation for the results per trial.

        Args:
            trial_index: Index of the trial.
        """

        self.stats.x_hat_sample_std[trial_index] = np.std(self.x_hat[trial_index], axis=0, ddof=1)
        self.stats.alpha_sample_std[trial_index] = np.std(self.alpha[trial_index], axis=0, ddof=1)

        self.stats.x_bar_sample_std[trial_index] = np.std(self.x_bar[trial_index], axis=0, ddof=1)
        self.stats.rho_sample_std[trial_index] = np.std(self.rho[trial_index], axis=0, ddof=1)

        self.stats.x_sample_std[trial_index] = np.std(self.x[trial_index], axis=0, ddof=1)
        self.stats.gamma_sample_std[trial_index] = np.std(self.gamma[trial_index], axis=0, ddof=1)

    def run(self):

        for e in range(self.num_trials):
            self.run_sim(e)

            self.compute_sample_mean(e)

            self.compute_sample_std(e)

            self.reset_agents()

        # @todo: clean this up to its own function
        self.stats.x = self.x
        self.stats.gamma = self.gamma

    def run_sim(self, trial_index):

        # Generate tiles (bernoulli instances for each agent)
        self.sim_data.tiles[trial_index] = self.generate_tiles(self.sim_data.num_agents)

        # Need period timing condition here to decide when to switch
        # between observation and communication
        curr_iteration = 0

        # Temporary data storage
        local_obs = []
        local_avg_black_obs = []

        local_x = []
        local_conf = []

        social_x = []
        social_conf = []

        informed_x = []
        informed_conf = []

        # Obtain initial estimates (before any observations are made)
        l, s, i = self.get_initial_estimates()

        local_x.append(l["x"])
        local_conf.append(l["conf"])

        social_x.append(s["x"])
        social_conf.append(s["conf"])

        informed_x.append(i["x"])
        informed_conf.append(i["conf"])

        # Go through each tile observation
        while curr_iteration < self.num_steps:

            # Execute observation phase
            local_obs_dict, local_val_dict = self.run_observation_phase(self.sim_data.tiles[trial_index][:, curr_iteration])

            local_obs.append(local_obs_dict["curr_obs"])
            local_avg_black_obs.append(local_obs_dict["avg_black_obs"])

            local_x.append(local_val_dict["x"])
            local_conf.append(local_val_dict["conf"])

            # Execute communication phase
            if (curr_iteration+1) % self.sim_data.comms_period == 0:
                social_val_dict = self.run_communication_phase()

                social_x.append(social_val_dict["x"])
                social_conf.append(social_val_dict["conf"])

            # Execute self computation phase
            informed_val_dict = self.run_self_computation_phase()

            informed_x.append(informed_val_dict["x"])
            informed_conf.append(informed_val_dict["conf"])

            curr_iteration += 1

        # Store observations and average black tile observations into log
        self.sim_data.agent_obs[trial_index] = np.asarray(local_obs).T

        # Store local estimates and confidences into log
        self.x_hat[trial_index] = np.asarray(local_x).T
        self.alpha[trial_index] = np.asarray(local_conf).T

        # Store social estimates and confidences into log
        self.x_bar[trial_index] = np.asarray(social_x).T
        self.rho[trial_index] = np.asarray(social_conf).T

        # Store informed estimates and confidences into log
        self.x[trial_index] = np.asarray(informed_x).T
        self.gamma[trial_index] = np.asarray(informed_conf).T

        # Ensure correctness
        assert self.sim_data.agent_obs[trial_index].shape[0] == self.sim_data.num_agents
        assert self.sim_data.agent_obs[trial_index].shape[1] == self.sim_data.num_steps

        assert self.x_hat[trial_index].shape[0] == self.sim_data.num_agents
        assert self.x_hat[trial_index].shape[1] == self.sim_data.num_steps + 1
        assert self.alpha[trial_index].shape[0] == self.sim_data.num_agents
        assert self.alpha[trial_index].shape[1] == self.sim_data.num_steps + 1

        assert self.x_bar[trial_index].shape[0] == self.sim_data.num_agents
        assert self.x_bar[trial_index].shape[1] == self.sim_data.num_steps//self.sim_data.comms_period + 1
        assert self.rho[trial_index].shape[0] == self.sim_data.num_agents
        assert self.rho[trial_index].shape[1] == self.sim_data.num_steps//self.sim_data.comms_period + 1

        assert self.x[trial_index].shape[0] == self.sim_data.num_agents
        assert self.x[trial_index].shape[1] == self.sim_data.num_steps + 1
        assert self.gamma[trial_index].shape[0] == self.sim_data.num_agents
        assert self.gamma[trial_index].shape[1] == self.sim_data.num_steps + 1

    def get_initial_estimates(self):
        """Get the initial estimates for all agents.

        Returns:
            A dict each containing a record of local estimates, social estimates, and informed estimates.
        """

        local_values = {"x": [], "conf": []}

        # Iterate through each agent to collect local estimate
        for v in self.sim_data.comms_network.graph.vertices():

            agent = self.sim_data.comms_network.agents_vp[v]

            # Collect local agent values
            local_values["x"].append(agent.get_x_hat())
            local_values["conf"].append(agent.get_alpha())

        # Collect social and informed estimates
        social_values = self.run_communication_phase()

        informed_values = self.run_self_computation_phase()

        return local_values, social_values, informed_values

    def run_communication_phase(self):
        """Execute one round of communication for all agents.

        Returns:
            A dict containing a record of social estimations and confidences.
        """

        social_values = {"x": [], "conf": []}

        # Make agents communicate
        for e in self.sim_data.comms_network.graph.edges():

            # Apply probabilistic communication (currently only applicable to undirected graphs)
            if random.random() < self.sim_data.comms_network.comms_prob_ep[e]:
                a1 = self.sim_data.comms_network.agents_vp[e.source()]
                a2 = self.sim_data.comms_network.agents_vp[e.target()]

                a1.communicate_rx( a2.communicate_tx() )
                a2.communicate_rx( a1.communicate_tx() )

        # Make the agents perform social (dual) and informed computation
        for v in self.sim_data.comms_network.graph.vertices():

            agent = self.sim_data.comms_network.agents_vp[v]

            agent.solve_social()

            social_values["x"].append(agent.get_x_bar())
            social_values["conf"].append(agent.get_rho())

        return social_values

    def run_observation_phase(self, tiles):
        """Execute one round of observation for all agents.

        Each agents will observe one tile from their observation reel.

        Returns:
            A dict containing a record of observations and a dict containing a record of
            agent estimations after local observations.
        """

        local_obs = {"curr_obs": [], "avg_black_obs": []}
        local_values = {"x": [], "conf": []}

        # Iterate through each agent to observe
        for ind, v in enumerate(self.sim_data.comms_network.graph.vertices()):

            agent = self.sim_data.comms_network.agents_vp[v]

            agent.observe(tiles[ind], self.observe_color)

            # Collect local agent values
            local_obs["curr_obs"].append(agent.get_curr_obs())
            local_obs["avg_black_obs"].append(agent.get_avg_black_obs())

            local_values["x"].append(agent.get_x_hat())
            local_values["conf"].append(agent.get_alpha())

        return local_obs, local_values

    def run_self_computation_phase(self):
        """Execute one round of self computation for all agents.
        """

        informed_values = {"x": [], "conf": []}

        # Make the agents perform social (dual) and informed computation
        for v in self.sim_data.comms_network.graph.vertices():

            agent = self.sim_data.comms_network.agents_vp[v]

            agent.solve_informed()

            informed_values["x"].append(agent.get_x())
            informed_values["conf"].append(agent.get_gamma())

        return informed_values

    def reset_agents(self):

        # Reset agents
        for v in self.sim_data.comms_network.graph.vertices():
            self.sim_data.comms_network.agents_vp[v].reset()

class CommsNetwork:
    """Class to wrap store communication network attributes.

    This class was specifically created to circumvent the complications related to serializing
    the graph-tool objects with internalized property maps. Instead, the property maps are simply
    set as the class attributes here along with the graph object.
    """

    def __init__(self, input_graph):
        self.graph = input_graph
        self.agents_vp = [] # agents vertex property
        self.comms_prob_ep = [] # communications probability edge property

class Agent:

    def __init__(self, p_b_b, p_w_w, local_functions, social_functions, informed_functions, legacy=False):

        if isinstance(p_b_b, list):
            self.prob_counter = 0
            self.b_prob_lst = p_b_b
            self.w_prob_lst = p_w_w

            self.b_prob = self.b_prob_lst[self.prob_counter]
            self.w_prob = self.w_prob_lst[self.prob_counter]
        else:
            self.b_prob = p_b_b
            self.w_prob = p_w_w

        self.total_b_tiles_obs = 0
        self.total_obs = 0

        self.comms_round_collected_est = []
        self.comms_round_collected_conf = []

        self.local_solver = self.LocalSolver(*local_functions)
        self.social_solver = self.SocialSolver(*social_functions, legacy=legacy)
        self.informed_solver = self.InformedSolver(*informed_functions)

    def reset(self):

        if hasattr(self, "prob_counter") and self.prob_counter+1 < len(self.b_prob_lst): # for distributed sensor probabilities
            self.prob_counter += 1
            self.b_prob = self.b_prob_lst[self.prob_counter]
            self.w_prob = self.w_prob_lst[self.prob_counter]

        self.total_b_tiles_obs = 0
        self.total_obs = 0

        self.comms_round_collected_est = []
        self.comms_round_collected_conf = []

        self.local_solver.reset()
        self.social_solver.reset()
        self.informed_solver.reset()

    def observe(self, encounter, obs_function):

        # Check if the encounter is black or white
        if encounter == WHITE_TILE:
            sensor_prob = self.w_prob
        elif encounter == BLACK_TILE:
            sensor_prob = self.b_prob
        else: # error catching
            raise RuntimeError("Error: invalid tile encounter!")

        # Record sensor observation of the encounter
        obs = obs_function(encounter, sensor_prob)

        self.curr_obs = obs
        self.total_b_tiles_obs += obs
        self.total_obs += 1

        # Update local estimation
        self.solve_local()

    def communicate_tx(self):
        """Transmit estimates and confidences.

        Returns:
            A tuple of estimate and confidence
        """

        return (self.local_solver.x, self.local_solver.conf)

    def communicate_rx(self, packet):
        """Receive estimates and confidences.

        Args:
            packet: A tuple of estimate and confdence.
        """

        self.comms_round_collected_est.append(packet[0])
        self.comms_round_collected_conf.append(packet[1])

    def solve_local(self):
        """Compute the local estimate and confidence.
        """
        self.local_solver.solve(self.total_b_tiles_obs,
                                self.total_obs,
                                self.b_prob,
                                self.w_prob)

    def solve_social(self):
        """Compute the social estimate and confidence.
        """
        self.social_solver.solve(self.comms_round_collected_est, self.comms_round_collected_conf)

        # Clear collected estimates and confidences since social estimate is computed
        self.comms_round_collected_est = []
        self.comms_round_collected_conf = []

    def solve_informed(self):
        """Compute the final estimate using the informed function.
        """
        self.informed_solver.solve(self.local_solver.x, self.local_solver.conf,
                                   self.social_solver.x, self.social_solver.conf)

    def get_x_hat(self): return self.local_solver.x

    def get_alpha(self): return self.local_solver.conf

    def get_x_bar(self): return self.social_solver.x

    def get_rho(self): return self.social_solver.conf

    def get_x(self): return self.informed_solver.x

    def get_gamma(self): return self.informed_solver.conf

    def get_curr_obs(self): return self.curr_obs

    def get_avg_black_obs(self): return self.total_b_tiles_obs / self.total_obs

    class LocalSolver:

        def __init__(self, est_func, conf_func):
            self.x = 0.5 # initial estimate is random
            self.conf = 0.0

            self.est_func = est_func
            self.conf_func = conf_func

        def reset(self):
            self.x = 0.5
            self.conf = 0.0

        def solve(self, h, t, b, w):
            """Compute the local estimate and confidence.
            """

            # Compute the local estimate
            self.x = self.est_func(h, t, b, w)

            # Compute the local confidence
            self.conf = self.conf_func(h, t, b, w)

    class SocialSolver:

        def __init__(self, est_func, conf_func, legacy=False):
            self.x = 0.0
            self.conf = 0.0

            self.est_func = est_func
            self.conf_func = conf_func

            self.legacy = legacy

        def reset(self):
            self.x = 0.0
            self.conf = 0.0

        def solve(self, x_arr, conf_arr):
            """Compute the social estimate and confidence.
            """

            # Compute the social estimate
            self.x = self.est_func(x_arr, conf_arr, self.legacy)

            # Compute the social confidence
            self.conf = self.conf_func(conf_arr, self.legacy)

    class InformedSolver:

        def __init__(self, primal_est_func, primal_conf_func):
            self.x = 0.0
            self.conf = 0.0

            self.primal_est_func = primal_est_func
            self.primal_conf_func = primal_conf_func

        def reset(self):
            self.x = 0.0
            self.conf = 0.0

        def solve(self, local_est, local_conf, social_est, social_conf):
            """Compute the combined estimate.
            """

            # Compute the informed estimate
            self.x = self.primal_est_func(local_est, local_conf, social_est, social_conf)
            self.conf = self.primal_conf_func(local_conf, social_conf)

class ExperimentData:
    """Class to store simulated experiment data.

    An instance of this class stores multi-agent simulation data that span all the desired
    fill ratio range and the desired sensor probabilities.
    """

    def __init__(self, sim_param_obj):
        self.sim_type = "static"
        self.num_agents = sim_param_obj.num_agents
        self.num_trials = sim_param_obj.num_trials
        self.num_steps = sim_param_obj.num_steps
        self.graph_type = sim_param_obj.comms_graph_str
        self.comms_period = sim_param_obj.comms_period
        self.comms_prob = sim_param_obj.comms_prob
        self.tfr_range = sim_param_obj.tfr_range
        self.sp_range = sim_param_obj.sp_range
        self.stats_obj_dict = {i: {j: None for j in self.sp_range} for i in self.tfr_range}
        self.sim_data_obj_dict = {i: {j: None for j in self.sp_range} for i in self.tfr_range}

    def insert_sim_obj(self, targ_fill_ratio, sensor_prob, stats_obj: Sim.SimStats, sim_data_obj: Sim.SimData):

        self.stats_obj_dict[targ_fill_ratio][sensor_prob] = stats_obj
        self.sim_data_obj_dict[targ_fill_ratio][sensor_prob] = sim_data_obj

    def get_stats_obj(self, targ_fill_ratio, sensor_prob) -> Sim.SimStats:
        """Get the statistics based on the target fill ratio and sensor probability.

        Args:
            targ_fill_ratio: Target fill ratio.
            sensor_prob: Sensor probability.

        Returns:
            A Sim.SimStats object containing the statistics for the simulation based on the target fill ratio and sensor probability.
        """

        return self.stats_obj_dict[targ_fill_ratio][sensor_prob]

    def get_sim_data_obj(self, targ_fill_ratio, sensor_prob) -> Sim.SimData:
        """Get the simulation data based on the target fill ratio and sensor probability.

        Args:
            targ_fill_ratio: Target fill ratio.
            sensor_prob: Sensor probability.

        Returns:
            A Sim.SimData object containing the data for the simulation based on the target fill ratio and sensor probability.
        """

        return self.sim_data_obj_dict[targ_fill_ratio][sensor_prob]

    # TODO: protobuf maybe? since the argos implementation will need that?
    def save(self, curr_time=None, filepath=None):
        """Serialize and save data.
        """

        # Get current time
        if curr_time is None:
            curr_time = datetime.now().strftime("%m%d%y_%H%M%S")

        if filepath:
            save_path = filepath + "_" + curr_time + ".ped"
        else:
            save_path = "multi_agent_sim_data_" + curr_time + ".ped"

        with open(save_path, "wb") as fopen:
            pickle.dump(self, fopen, pickle.HIGHEST_PROTOCOL)

        print( "\nSaved multi-agent sim data at: {0}.\n".format( os.path.abspath(save_path) ) )

    @classmethod
    def load(cls, filepath, debug=False):
        """Load serialized data.
        """

        with open(filepath, 'rb') as fopen:
            obj = pickle.load(fopen)

        # Verify the unpickled object
        assert isinstance(obj, cls)

        # Remove objects to save RAM
        if debug:
            obj.stats_obj_dict = {}
        else:
            obj.sim_data_obj_dict = {}

        return obj

class SimParam:

    def __init__(self, yaml_config):

        # Common parameters
        tfr_min = float(yaml_config["targFillRatios"]["min"])
        tfr_max = float(yaml_config["targFillRatios"]["max"])
        tfr_inc = int(yaml_config["targFillRatios"]["incSteps"])

        sp_min = float(yaml_config["sensorProb"]["min"])
        sp_max = float(yaml_config["sensorProb"]["max"])
        sp_inc = int(yaml_config["sensorProb"]["incSteps"])

        # Check if distributed sensor probabilities is desired
        if sp_inc < 0:
            self.sp_range = encode_sp_distribution(sp_inc, sp_min, sp_max)
        else:
            self.sp_range = np.round(np.linspace(sp_min, sp_max, sp_inc), 3).tolist()
        self.tfr_range = np.round(np.linspace(tfr_min, tfr_max, tfr_inc), 3).tolist()
        self.num_steps = int(yaml_config["numSteps"])
        self.num_trials = int(yaml_config["numTrials"])
        self.legacy = yaml_config["legacy"]

        # Multi-agent simulation parameters
        try:
            self.num_agents = int(yaml_config["numAgents"])
            self.comms_graph_str = yaml_config["commsGraph"]["type"]
            self.comms_period = int(yaml_config["commsGraph"]["commsPeriod"])
            self.comms_prob = float(yaml_config["commsGraph"]["commsProb"])
        except Exception as e:
            pass

        self.create_filename_descriptors()

    def create_filename_descriptors(self):
        """Create filename descriptors for informative folder and filenames.
        """

        # Compute increment step sizes
        if len(self.sp_range) == 1:
            sensor_prob_inc = 0
        else:
            sensor_prob_inc = self.sp_range[1] - self.sp_range[0]

        if len(self.tfr_range) == 1:
            fill_ratio_inc = 0
        else:
            fill_ratio_inc = self.tfr_range[1] - self.tfr_range[0]

        # Define filename descriptors
        min_sensor_prob, max_sensor_prob = int(self.sp_range[0]*1e2), int(self.sp_range[-1]*1e2)
        min_targ_fill_ratio, max_targ_fill_ratio = int(self.tfr_range[0]*1e2), int(self.tfr_range[-1]*1e2)
        p_inc, f_inc = int(sensor_prob_inc*1e2), int(fill_ratio_inc*1e2)

        self.filename_suffix_1 = "_e" +str(self.num_trials) + "_o" + str(self.num_steps) # describing number of trials and observations

        prob_suffix = "_p" + str(min_sensor_prob) + "-" + str(p_inc) + "-" + str(max_sensor_prob)
        f_suffix = "_f" + str(min_targ_fill_ratio) + "-" + str(f_inc) + "-" + str(max_targ_fill_ratio)
        self.filename_suffix_2 = prob_suffix + f_suffix # describing the probabilites and fill ratios

        self.full_suffix = self.filename_suffix_1 + self.filename_suffix_2