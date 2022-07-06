import yaml
import numpy as np
import csv
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

class Sim:
    """Top level simulation class.
    """

    class SimData:
        """Class for storing simulation data that can be used replicate experimental results.
        """

        def __init__(self, sim_type, num_exp, num_agents, num_obs, sensor_prob, comms_period):
            self.sim_type = sim_type
            self.num_exp = num_exp
            self.num_agents = num_agents
            self.num_obs = num_obs
            self.b_prob = sensor_prob # P(black|black)
            self.w_prob = sensor_prob # P(white|white)
            self.comms_network_str = None
            self.comms_period = comms_period

            if sim_type == "single":
                self.tiles = np.zeros( (num_exp, num_obs) )
                self.agent_obs = np.zeros( (num_exp, num_obs) )

            elif sim_type == "multi":
                self.tiles = np.zeros( (num_exp, num_agents, num_obs) )
                self.agent_obs = np.zeros( (num_exp, num_agents, num_obs) )

    class SimStats:
        """Class for storing statistics of simulation experiments.
        """

        def __init__(self, sim_type, num_exp=0, num_obs=0, comms_period=1, num_agents=0):

            if sim_type == "single" and num_exp != 0 and num_obs != 0:
                self.x_hat_sample_mean = np.zeros( (num_exp, num_obs + 1) )
                self.alpha_sample_mean = np.zeros( (num_exp, num_obs + 1) )
                self.x_hat_sample_std = np.zeros( (num_exp, num_obs + 1) )
                self.alpha_sample_std = np.zeros( (num_exp, num_obs + 1) )

            elif sim_type == "multi" and num_exp != 0 and num_obs != 0:
                self.sp_distribution = None
                self.sp_distributed_sample_mean = np.zeros(num_exp)

                # @todo: temporary hack to show individual robot values; in the future this should be stored
                # elsewhere
                self.x = np.zeros( (num_exp, num_agents, num_obs//comms_period + 1) )
                self.gamma = np.zeros( (num_exp, num_agents, num_obs//comms_period + 1) )

                self.x_hat_sample_mean = np.zeros( (num_exp, num_obs + 1) )
                self.alpha_sample_mean = np.zeros( (num_exp, num_obs + 1) )
                self.x_hat_sample_std = np.zeros( (num_exp, num_obs + 1) )
                self.alpha_sample_std = np.zeros( (num_exp, num_obs + 1) )

                self.x_bar_sample_mean = np.zeros( (num_exp, num_obs//comms_period + 1) )
                self.rho_sample_mean = np.zeros( (num_exp, num_obs//comms_period + 1) )
                self.x_bar_sample_std = np.zeros( (num_exp, num_obs//comms_period + 1) )
                self.rho_sample_std = np.zeros( (num_exp, num_obs//comms_period + 1) )

                self.x_sample_mean = np.zeros( (num_exp, num_obs//comms_period + 1) )
                self.gamma_sample_mean = np.zeros( (num_exp, num_obs//comms_period + 1) )
                self.x_sample_std = np.zeros( (num_exp, num_obs//comms_period + 1) )
                self.gamma_sample_std = np.zeros( (num_exp, num_obs//comms_period + 1) )

            else: # for population with dynamic simulation data
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

    def __init__(self, num_exp, num_obs, des_fill_ratio, main_filename_suffix):
        self.num_exp = num_exp
        self.num_obs = num_obs
        self.des_fill_ratio = des_fill_ratio
        self.avg_fill_ratio = 0.0
        self.tiles_record = np.empty( (self.num_exp, self.num_obs) )
        self.main_filename_suffix = main_filename_suffix

    def generate_tiles(self, num_agents=1):
        """Generate the tiles based on the desired/nominal fill ratios.

        Args:
            num_agents: Number of tiles rows to generate, only used in multi-agent simulations.

        Returns:
            A 1-D (single agent simulation) or a (num_agents x self.num_obs) 2-D numpy array
             of binary tiles (multi-agent simulation).
        """

        # Draw bernoulli samples for tiles based on desired fill ratio
        if num_agents > 1:
            tiles = np.random.binomial(1, self.des_fill_ratio * np.ones((num_agents, self.num_obs)))

        else:
            tiles = np.random.binomial(1, self.des_fill_ratio * np.ones(self.num_obs) )

            assert(len(tiles) == self.num_obs)

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

    def compute_fisher_hat_inv(self, h, t, b, w):
        """Compute the inverse Fisher information (variance) for one agent.
        """

        if (b == 1.0) and (w == 1.0): output = h * (t-h) / np.power(t, 3) # perfect sensors
        else: # noisy sensors
            if h <= (1.0 - w) * t:
                output = np.square(w) * np.square(w-1.0) / ( np.square(b+w-1.0) * (t*np.square(w) - 2*(t-h)*w + (t-h)) )
            elif h >= b*t:
                output = np.square(b) * np.square(b-1.0) / ( np.square(b+w-1.0) * (t*np.square(b) - 2*h*b + h) )
            else:
                output = h * (t-h) / ( np.power(t, 3) * np.square(b+w-1.0) )

        return np.nan_to_num(output, posinf=POSINF) # prevent nans

    def compute_fisher_hat(self, h, t, b, w):
        """Compute the Fisher information for one agent.
        """
        return np.nan_to_num( np.reciprocal( self.compute_fisher_hat_inv(h, t, b, w) ), posinf=POSINF ) # prevent infs

    def compute_x_bar(self, x_arr): # TODO: need to split this out of the parent class since it should be modular (i.e., we may not use the same social function)
        return np.mean(x_arr)

    def compute_fisher_bar(self, fisher_arr):
        return np.nan_to_num(np.reciprocal( np.mean( np.reciprocal( fisher_arr ) ) ), posinf=POSINF) # prevent infs

    def compute_x(self, x_hat, alpha, x_bar, rho): # TODO: need to split this out of the parent class since it should be modular (i.e., we may not use the same objective function)
        return ( alpha*x_hat + rho*x_bar ) / (alpha + rho)

    def compute_fisher(self, alpha, rho):
        return np.nan_to_num(np.reciprocal( 1 / (alpha + rho) ), posinf=POSINF) # alpha^-1 and rho^-1 are variances; prevent infs

    def _write_data_to_csv(self, f_hat_data, fisher_inv_data, suffix=""):

        f_hat_filename = "f_hat" + self.main_filename_suffix + suffix + ".csv"
        fisher_inv_filename = "fisher_inv" + self.main_filename_suffix + suffix + ".csv"
        tiles_filename =  "tiles" + self.main_filename_suffix + suffix + ".csv"

        with open(f_hat_filename, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(f_hat_data)

        with open(fisher_inv_filename, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(fisher_inv_data)

        with open(tiles_filename, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.tiles_record)

class SingleAgentSim(Sim):

    def __init__(self, num_exp, num_obs, des_fill_ratio, b_prob, w_prob, main_f_suffix):

        super().__init__(num_exp, num_obs, des_fill_ratio, main_f_suffix)

        self.b_prob = b_prob # P(black|black)
        self.w_prob = w_prob # P(white|white)
        self.agent_obs = np.zeros( (num_exp, num_obs) )
        self.agent_avg_black_obs = np.zeros( (num_exp, num_obs) )

        # Define data members
        self.f_hat = np.zeros( (num_exp, num_obs) )
        self.fisher_inv = np.zeros( (num_exp, num_obs) )
        self.f_hat_sample_mean = np.zeros( num_obs )
        self.fisher_inv_sample_mean = np.zeros( num_obs )
        self.f_hat_sample_std = np.zeros( num_obs )
        self.fisher_inv_sample_std = np.zeros( num_obs )
        self.f_hat_sample_min = np.zeros( num_obs )
        self.fisher_inv_sample_min = np.zeros( num_obs )
        self.f_hat_sample_max = np.zeros( num_obs )
        self.fisher_inv_sample_max = np.zeros( num_obs )

    def run(self, data_flag = False):
        self.run_sim()

        self.compute_sample_mean()

        self.compute_sample_std()

        self.compute_sample_min()

        self.compute_sample_max()

        if data_flag: self.write_data_to_csv()

    def compute_sample_mean(self):

        self.f_hat_sample_mean = np.mean(self.f_hat, axis=0)
        self.fisher_inv_sample_mean = np.mean(self.fisher_inv, axis=0)

    def compute_sample_std(self):

        # TODO: should i be using the biased or unbiased variance here? sticking with the unbiased for now
        self.f_hat_sample_std = np.std(self.f_hat, axis=0, ddof=1)
        self.fisher_inv_sample_std = np.std(self.fisher_inv, axis=0, ddof=1)

    def compute_sample_min(self):

        self.f_hat_sample_min = np.amin(self.f_hat, axis=0)
        self.fisher_inv_sample_min = np.amin(self.fisher_inv, axis=0)

    def compute_sample_max(self):

        self.f_hat_sample_max = np.amax(self.f_hat, axis=0)
        self.fisher_inv_sample_max = np.amax(self.fisher_inv, axis=0)

    def run_sim(self):

        # Compute the denominator term for the estimated fill ratio calculation
        denom = self.b_prob + self.w_prob - 1.0

        for exp_ind in range(self.num_exp):
            prev_obs = 0
            curr_obs = 0

            tiles = self.generate_tiles() # generate a new set of tiles for the current agent/experiment

            for tile_ind, tile_color in enumerate(tiles):

                if tile_color == 1:
                    curr_obs = self.observe_color(tile_color, self.b_prob)
                else:
                    curr_obs = self.observe_color(tile_color, self.w_prob)

                # Store observations
                self.agent_obs[exp_ind][tile_ind] = curr_obs
                self.agent_avg_black_obs[exp_ind][tile_ind] = (prev_obs + curr_obs) / (tile_ind + 1) # tile number in the denom
                prev_obs += curr_obs

                # Compute estimated fill ratio
                if self.agent_avg_black_obs[exp_ind][tile_ind] <= 1 - self.w_prob:
                    self.f_hat[exp_ind][tile_ind] = 0.0
                elif self.agent_avg_black_obs[exp_ind][tile_ind] >= self.b_prob:
                    self.f_hat[exp_ind][tile_ind] = 1.0
                else:
                    self.f_hat[exp_ind][tile_ind] = (self.agent_avg_black_obs[exp_ind][tile_ind] + self.w_prob - 1.0) / denom

                # Compute the inverse Fisher information (variance)
                h = np.sum(self.agent_obs[exp_ind][0:tile_ind+1])
                self.fisher_inv[exp_ind][tile_ind] = self.compute_fisher_hat_inv(h, tile_ind + 1, self.b_prob, self.w_prob)

            # Store the tile config
            self.tiles_record[exp_ind] = tiles

            # Compute the average tile ratio up to this simulation experiment
            self.avg_fill_ratio =  (exp_ind)/(exp_ind+1) * self.avg_fill_ratio + 1/(exp_ind+1) * sum(tiles)/self.num_obs

    def write_data_to_csv(self):
        """Write simulation data to CSV files.
        """

        prob_suffix = "_b" + str( int(self.b_prob*1e2) ) + "w" + str( int(self.w_prob*1e2) )
        f_suffix = "_df" + str( int(self.des_fill_ratio*1e2) ) + "af" + str( int(self.avg_fill_ratio*1e2) )
        suffix = prob_suffix + f_suffix

        self._write_data_to_csv( self.f_hat, self.fisher_inv, suffix )

class MultiAgentSim(Sim):

    def __init__(self, sim_param_obj, des_fill_ratio, sensor_prob):

        num_agents = sim_param_obj.num_agents
        num_exp = sim_param_obj.num_exp
        num_obs = sim_param_obj.num_obs
        comms_period = sim_param_obj.comms_period

        super().__init__(num_exp, num_obs, des_fill_ratio, sim_param_obj.filename_suffix_1)

        # Initialize data containers (to be serialized)
        self.stats = self.SimStats("multi", num_exp, num_obs, comms_period, num_agents)
        if sensor_prob < 0: # not actually the sensor probability; actually encoded distribution

            # Decode distribution parameters
            val = int( str(sensor_prob)[:2] )
            param_1 = float( str( sensor_prob )[2:6] ) * 1e-3
            param_2 = float( str( sensor_prob )[6:] ) * 1e-3

            # Store parameters
            self.generator = np.random.default_rng()
            self.dist_params = [param_1, param_2]
            self.sim_data = self.SimData("multi", num_exp, num_agents, num_obs, val, comms_period)
        else:
            self.sim_data = self.SimData("multi", num_exp, num_agents, num_obs, sensor_prob, comms_period)

        # Initialize non-persistent simulation data
        self.x_hat = np.zeros( (num_exp, num_agents, num_obs + 1) )
        self.alpha = np.zeros( (num_exp, num_agents, num_obs + 1) )
        self.x_bar = np.zeros( (num_exp, num_agents, num_obs//comms_period + 1) )
        self.rho = np.zeros( (num_exp, num_agents, num_obs//comms_period + 1) )
        self.x = np.zeros( (num_exp, num_agents, num_obs//comms_period + 1) )
        self.gamma = np.zeros( (num_exp, num_agents, num_obs//comms_period + 1) )

        # Setup up communication graph
        self.setup_comms_graph(sim_param_obj.comms_graph_str, sim_param_obj.comms_prob)
        self.create_agents()

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

    def create_agents(self):

        # Create agent objects
        agents_vprop = self.sim_data.comms_network.graph.new_vertex_property("object") # need to populate agents into the vertices
        sensor_probs = []

        for vertex in self.sim_data.comms_network.graph.get_vertices():
            if self.sim_data.b_prob == UNIFORM_DIST_SP_ENUM:
                b_sensor_prob = ( (self.dist_params[1] - self.dist_params[0]) * self.generator.random(self.num_exp) + self.dist_params[0] ).tolist()
                w_sensor_prob = b_sensor_prob

                sensor_probs.append(b_sensor_prob)

                self.stats.sp_distribution = "uniform"
                dist_function = self.generator.random()

            elif self.sim_data.b_prob == NORMAL_DIST_SP_ENUM:
                b_sensor_prob = ( self.generator.normal(self.dist_params[0], np.sqrt(self.dist_params[1]), self.num_exp) ).tolist()
                w_sensor_prob = b_sensor_prob

                sensor_probs.append(b_sensor_prob)

                self.stats.sp_distribution = "normal"
                dist_function = self.generator.normal()

            else:
                b_sensor_prob = self.sim_data.b_prob
                w_sensor_prob = self.sim_data.w_prob
                sensor_probs = [b_sensor_prob]

                dist_function = None

            agents_vprop[vertex] = Agent(b_sensor_prob, w_sensor_prob,
                                        (self.compute_x_hat, self.compute_fisher_hat),
                                        (self.compute_x_bar, self.compute_fisher_bar),
                                        (self.compute_x, self.compute_fisher))

        # @todo hacky solution to replace/update mean sensor probability
        if self.stats.sp_distribution:
            self.sim_data.b_prob = sensor_probs
            self.sim_data.w_prob = sensor_probs

        self.stats.sp_distributed_sample_mean = np.mean(sensor_probs, axis=0)

        self.sim_data.comms_network.agents_vp = agents_vprop

    def compute_sample_mean(self, experiment_index):
        """Compute the sample mean for the results per experiment.

        Args:
            experiment_index: Index of the experiment.
        """

        self.stats.x_hat_sample_mean[experiment_index] = np.mean(self.x_hat[experiment_index], axis=0)
        self.stats.alpha_sample_mean[experiment_index] = np.mean(self.alpha[experiment_index], axis=0)

        self.stats.x_bar_sample_mean[experiment_index] = np.mean(self.x_bar[experiment_index], axis=0)
        self.stats.rho_sample_mean[experiment_index] = np.mean(self.rho[experiment_index], axis=0)

        self.stats.x_sample_mean[experiment_index] = np.mean(self.x[experiment_index], axis=0)
        self.stats.gamma_sample_mean[experiment_index] = np.mean(self.gamma[experiment_index], axis=0)

    def compute_sample_std(self, experiment_index):
        """Compute the sample standard deviation for the results per experiment.

        Args:
            experiment_index: Index of the experiment.
        """

        self.stats.x_hat_sample_std[experiment_index] = np.std(self.x_hat[experiment_index], axis=0, ddof=1)
        self.stats.alpha_sample_std[experiment_index] = np.std(self.alpha[experiment_index], axis=0, ddof=1)

        self.stats.x_bar_sample_std[experiment_index] = np.std(self.x_bar[experiment_index], axis=0, ddof=1)
        self.stats.rho_sample_std[experiment_index] = np.std(self.rho[experiment_index], axis=0, ddof=1)

        self.stats.x_sample_std[experiment_index] = np.std(self.x[experiment_index], axis=0, ddof=1)
        self.stats.gamma_sample_std[experiment_index] = np.std(self.gamma[experiment_index], axis=0, ddof=1)

    def run(self):

        for e in range(self.num_exp):
            self.run_sim(e)

            self.compute_sample_mean(e)

            self.compute_sample_std(e)

            self.reset_agents()

        # @todo: clean this up to its own function
        self.stats.x = self.x
        self.stats.gamma = self.gamma

    def run_sim(self, experiment_index):

        # Generate tiles (bernoulli instances for each agent)
        self.sim_data.tiles[experiment_index] = self.generate_tiles(self.sim_data.num_agents)

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
        while curr_iteration < self.num_obs:

            # Execute observation phase
            local_obs_dict, local_val_dict = self.run_observation_phase(self.sim_data.tiles[experiment_index][:, curr_iteration])

            local_obs.append(local_obs_dict["curr_obs"])
            local_avg_black_obs.append(local_obs_dict["avg_black_obs"])

            local_x.append(local_val_dict["x"])
            local_conf.append(local_val_dict["conf"])

            # Execute communication phase
            if (curr_iteration+1) % self.sim_data.comms_period == 0:
                social_val_dict, informed_val_dict = self.run_communication_phase()

                social_x.append(social_val_dict["x"])
                social_conf.append(social_val_dict["conf"])

                informed_x.append(informed_val_dict["x"])
                informed_conf.append(informed_val_dict["conf"])

            curr_iteration += 1

        # Store observations and average black tile observations into log
        self.sim_data.agent_obs[experiment_index] = np.asarray(local_obs).T

        # Store local estimates and confidences into log
        self.x_hat[experiment_index] = np.asarray(local_x).T
        self.alpha[experiment_index] = np.asarray(local_conf).T

        # Store social estimates and confidences into log
        self.x_bar[experiment_index] = np.asarray(social_x).T
        self.rho[experiment_index] = np.asarray(social_conf).T

        # Store informed estimates and confidences into log
        self.x[experiment_index] = np.asarray(informed_x).T
        self.gamma[experiment_index] = np.asarray(informed_conf).T

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
        social_values, informed_values = self.run_communication_phase()

        return local_values, social_values, informed_values

    def run_communication_phase(self):
        """Execute one round of communication for all agents.

        Returns:
            A dict containing a record of social estimations and confidences 
            and a dict containing a record of informed estimations and confidences.
        """

        social_values = {"x": [], "conf": []}
        informed_values = {"x": [], "conf": []}

        # Make agents communicate
        for e in self.sim_data.comms_network.graph.edges():

            # Apply probabilistic communication (currently only applicable to undirected graphs)
            if random.random() < self.sim_data.comms_network.comms_prob_ep[e]:
                a1 = self.sim_data.comms_network.agents_vp[e.source()]
                a2 = self.sim_data.comms_network.agents_vp[e.target()]

                a1.communicate_rx( a2.communicate_tx() )
                a2.communicate_rx( a1.communicate_tx() )

        # Make the agents perform social (dual) and primal computation
        for v in self.sim_data.comms_network.graph.vertices():
            agent = self.sim_data.comms_network.agents_vp[v]

            agent.solve_social()
            agent.solve_primal()

            social_values["x"].append(agent.get_x_bar())
            social_values["conf"].append(agent.get_rho())
            informed_values["x"].append(agent.get_x())
            informed_values["conf"].append(agent.get_gamma())

        return social_values, informed_values

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

    def __init__(self, p_b_b, p_w_w, local_functions, social_functions, primal_functions):

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
        self.social_solver = self.SocialSolver(*social_functions)
        self.primal_solver = self.PrimalSolver(*primal_functions)

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
        self.primal_solver.reset()

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

    def solve_primal(self):
        """Compute the final estimate using the primal function.
        """
        self.primal_solver.solve(self.local_solver.x, self.local_solver.conf,
                                 self.social_solver.x, self.social_solver.conf)

    def get_x_hat(self): return self.local_solver.x

    def get_alpha(self): return self.local_solver.conf

    def get_x_bar(self): return self.social_solver.x

    def get_rho(self): return self.social_solver.conf

    def get_x(self): return self.primal_solver.x

    def get_gamma(self): return self.primal_solver.conf

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

        def __init__(self, est_func, conf_func):
            self.x = 0.0
            self.conf = 0.0

            self.est_func = est_func
            self.conf_func = conf_func

        def reset(self):
            self.x = 0.0
            self.conf = 0.0

        def solve(self, x_arr, conf_arr):
            """Compute the social estimate and confidence.
            """

            # Compute the social estimate
            self.x = self.est_func(x_arr)

            # Compute the social confidence
            self.conf = self.conf_func(conf_arr)

    class PrimalSolver:

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
        self.num_exp = sim_param_obj.num_exp
        self.num_obs = sim_param_obj.num_obs
        self.graph_type = sim_param_obj.comms_graph_str
        self.comms_period = sim_param_obj.comms_period
        self.comms_prob = sim_param_obj.comms_prob
        self.dfr_range = sim_param_obj.dfr_range
        self.sp_range = sim_param_obj.sp_range
        self.stats_obj_dict = {i: {j: None for j in self.sp_range} for i in self.dfr_range}
        self.sim_data_obj_dict = {i: {j: None for j in self.sp_range} for i in self.dfr_range}

    def insert_sim_obj(self, des_fill_ratio, sensor_prob, stats_obj: Sim.SimStats, sim_data_obj: Sim.SimData):

        self.stats_obj_dict[des_fill_ratio][sensor_prob] = stats_obj
        self.sim_data_obj_dict[des_fill_ratio][sensor_prob] = sim_data_obj

    def get_stats_obj(self, des_fill_ratio, sensor_prob) -> Sim.SimStats:
        """Get the statistics based on the target fill ratio and sensor probability.

        Args:
            des_fill_ratio: Target fill ratio.
            sensor_prob: Sensor probability.

        Returns:
            A Sim.SimStats object containing the statistics for the simulation based on the target fill ratio and sensor probability.
        """

        return self.stats_obj_dict[des_fill_ratio][sensor_prob]

    def get_sim_data_obj(self, des_fill_ratio, sensor_prob) -> Sim.SimData:
        """Get the simulation data based on the target fill ratio and sensor probability.

        Args:
            des_fill_ratio: Target fill ratio.
            sensor_prob: Sensor probability.

        Returns:
            A Sim.SimData object containing the data for the simulation based on the target fill ratio and sensor probability.
        """

        return self.sim_data_obj_dict[des_fill_ratio][sensor_prob]

    # TODO: protobuf maybe? since the argos implementation will need that?
    def save(self, curr_time=None, filepath=None):
        """Serialize and save data.
        """

        # Get current time
        if curr_time is None:
            curr_time = datetime.now().strftime("%m%d%y_%H%M%S")

        if filepath:
            root, ext = os.path.splitext(filepath)
            save_path = root + "_" + curr_time + ext
        else:
            save_path = "multi_agent_sim_data_" + curr_time + ".pkl"

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

class HeatmapData:
    """Class to store and process heatmap data
    """

    def __init__(self, sim_param_obj):

        self.num_exp = sim_param_obj.num_exp
        self.num_obs = sim_param_obj.num_obs
        self.sensor_prob_range = sim_param_obj.sp_range
        self.fill_ratio_range = sim_param_obj.dfr_range
        self.main_filename_suffix = sim_param_obj.full_suffix

        self.f_hat_data = {"mean": [], "min": [], "max": []}
        self.fisher_inv_data = {"mean": [], "min": [], "max": []}
        self.avg_f_data = []

    def compile_data(self, heatmap_row_obj):
        """Compile and organize heatmap data.
        """

        self.f_hat_data["mean"].append(heatmap_row_obj.f_hat_mean)
        self.f_hat_data["min"].append(heatmap_row_obj.f_hat_min)
        self.f_hat_data["max"].append(heatmap_row_obj.f_hat_max)
        self.fisher_inv_data["mean"].append(heatmap_row_obj.fisher_inv_mean)
        self.fisher_inv_data["min"].append(heatmap_row_obj.fisher_inv_min)
        self.fisher_inv_data["max"].append(heatmap_row_obj.fisher_inv_max)

        self.avg_f_data.append(heatmap_row_obj.avg_f)

    def write_data_to_csv(self):
        """Write completed heatmap data to CSV files.
        """

        # Write heatmap data
        f_hat_mean_filename = "f_hat_heatmap_mean" + self.main_filename_suffix + ".csv"
        f_hat_min_filename = "f_hat_heatmap_min" + self.main_filename_suffix + ".csv"
        f_hat_max_filename = "f_hat_heatmap_max" + self.main_filename_suffix + ".csv"
        fisher_inv_mean_filename = "fisher_inv_heatmap_mean" + self.main_filename_suffix + ".csv"
        fisher_inv_min_filename = "fisher_inv_heatmap_min" + self.main_filename_suffix + ".csv"
        fisher_inv_max_filename = "fisher_inv_heatmap_max" + self.main_filename_suffix + ".csv"
        des_f_avg_f_filename = "des_f_avg_f" + self.main_filename_suffix + ".csv"

        with open(f_hat_mean_filename, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.f_hat_data["mean"])

        with open(f_hat_min_filename, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.f_hat_data["min"])

        with open(f_hat_max_filename, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.f_hat_data["max"])

        with open(fisher_inv_mean_filename, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.fisher_inv_data["mean"])

        with open(fisher_inv_min_filename, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.fisher_inv_data["min"])

        with open(fisher_inv_max_filename, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.fisher_inv_data["max"])

        with open(des_f_avg_f_filename, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows( np.array( (self.fill_ratio_range, self.avg_f_data) ).T )

class HeatmapRow:
    """Class to store one row of heatmap data
    """

    def __init__(self):
        self.avg_f = 0.0

        self.f_hat_mean = []
        self.f_hat_min = []
        self.f_hat_max = []

        self.fisher_inv_mean = []
        self.fisher_inv_min = []
        self.fisher_inv_max = []

    # TODO: update variable names to use alpha instead of fisher_inv
    def populate(self, single_agent_sim_obj: SingleAgentSim):
        """Populate data given SingleAgentSim object.
        """

        self.f_hat_mean.append( single_agent_sim_obj.f_hat_sample_mean[-1] )
        self.f_hat_min.append( single_agent_sim_obj.f_hat_sample_min[-1] )
        self.f_hat_max.append( single_agent_sim_obj.f_hat_sample_max[-1] )

        self.fisher_inv_mean.append( single_agent_sim_obj.fisher_inv_sample_mean[-1] )
        self.fisher_inv_min.append( single_agent_sim_obj.fisher_inv_sample_min[-1] )
        self.fisher_inv_max.append( single_agent_sim_obj.fisher_inv_sample_max[-1] )

        self.avg_f = single_agent_sim_obj.avg_fill_ratio

    def populate(self, multi_agent_sim_obj: MultiAgentSim):

        # Collect local analytics
        self.x_hat_mean.append( multi_agent_sim_obj.x_hat_sample_mean[-1] )
        self.x_hat_min.append( multi_agent_sim_obj.x_hat_sample_min[-1] )
        self.x_hat_max.append( multi_agent_sim_obj.x_hat_sample_max[-1] )

        self.alpha_mean.append( multi_agent_sim_obj.alpha_sample_mean[-1] )
        self.alpha_min.append( multi_agent_sim_obj.alpha_sample_min[-1] )
        self.alpha_max.append( multi_agent_sim_obj.alpha_sample_max[-1] )

        # Collect social analytics
        self.x_bar_mean.append( multi_agent_sim_obj.x_bar_sample_mean[-1] )
        self.x_bar_min.append( multi_agent_sim_obj.x_bar_sample_min[-1] )
        self.x_bar_max.append( multi_agent_sim_obj.x_bar_sample_max[-1] )

        self.rho_mean.append( multi_agent_sim_obj.rho_sample_mean[-1] )
        self.rho_min.append( multi_agent_sim_obj.rho_sample_min[-1] )
        self.rho_max.append( multi_agent_sim_obj.rho_sample_max[-1] )

        # Collect informed analytics
        self.x_mean.append( multi_agent_sim_obj.x_sample_mean[-1] )
        self.x_min.append( multi_agent_sim_obj.x_sample_min[-1] )
        self.x_max.append( multi_agent_sim_obj.x_sample_max[-1] )

        self.gamma_mean.append( multi_agent_sim_obj.gamma_sample_mean[-1] )
        self.gamma_min.append( multi_agent_sim_obj.gamma_sample_min[-1] )
        self.gamma_max.append( multi_agent_sim_obj.gamma_sample_max[-1] )

        self.avg_f = multi_agent_sim_obj.avg_fill_ratio
        pass

class SimParam:

    def __init__(self, yaml_config):

        # Common parameters
        dfr_min = float(yaml_config["desFillRatios"]["min"])
        dfr_max = float(yaml_config["desFillRatios"]["max"])
        dfr_inc = int(yaml_config["desFillRatios"]["incSteps"])

        sp_min = float(yaml_config["sensorProb"]["min"])
        sp_max = float(yaml_config["sensorProb"]["max"])
        sp_inc = int(yaml_config["sensorProb"]["incSteps"])

        # Check if distributed sensor probabilities is desired
        if sp_inc == UNIFORM_DIST_SP_ENUM:
            # Encode distribution parameters into single int in a list
            lower_bound = "{:04d}".format( int( np.round( sp_min*1e3, 3 ) ) ) # 4 digits, scaled by 1e3
            upper_bound = "{:04d}".format( int( np.round( sp_max*1e3, 3 ) ) ) # 4 digits, scaled by 1e3
            self.sp_range = [ int( str(UNIFORM_DIST_SP_ENUM) + lower_bound + upper_bound ) ] # [distribution id, lower bound incl., upper bound excl.]

        elif sp_inc == NORMAL_DIST_SP_ENUM:
            # Encode distribution parameters into single int in a list
            mean = "{:04d}".format( int( np.round( sp_min*1e3, 3 ) ) ) # 4 digits, scaled by 1e3
            var = "{:04d}".format( int( np.round( sp_max*1e3, 3 ) ) ) # 4 digits, scaled by 1e3
            self.sp_range = [ int(str(NORMAL_DIST_SP_ENUM) + mean + var) ] # [distribution id, mean, variance]

        else:
            self.sp_range = np.round(np.linspace(sp_min, sp_max, sp_inc), 3).tolist()
        self.dfr_range = np.round(np.linspace(dfr_min, dfr_max, dfr_inc), 3).tolist()
        self.num_obs = int(yaml_config["numObs"])
        self.num_exp = int(yaml_config["numExperiments"])
        self.write_all = yaml_config["writeAllData"]

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

        if len(self.dfr_range) == 1:
            fill_ratio_inc = 0
        else:
            fill_ratio_inc = self.dfr_range[1] - self.dfr_range[0]

        # Define filename descriptors
        min_sensor_prob, max_sensor_prob = int(self.sp_range[0]*1e2), int(self.sp_range[-1]*1e2)
        min_des_fill_ratio, max_des_fill_ratio = int(self.dfr_range[0]*1e2), int(self.dfr_range[-1]*1e2)
        p_inc, f_inc = int(sensor_prob_inc*1e2), int(fill_ratio_inc*1e2)

        self.filename_suffix_1 = "_e" +str(self.num_exp) + "_o" + str(self.num_obs) # describing number of experiments and observations

        prob_suffix = "_p" + str(min_sensor_prob) + "-" + str(p_inc) + "-" + str(max_sensor_prob)
        f_suffix = "_f" + str(min_des_fill_ratio) + "-" + str(f_inc) + "-" + str(max_des_fill_ratio)
        self.filename_suffix_2 = prob_suffix + f_suffix # describing the probabilites and fill ratios

        self.full_suffix = self.filename_suffix_1 + self.filename_suffix_2