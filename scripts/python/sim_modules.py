import numpy as np
import csv
from datetime import datetime

class Sim:

    def __init__(self, num_cycles, num_obs, fill_ratio):
        self.num_cycles = num_cycles
        self.num_obs = num_obs
        self.fill_ratio = fill_ratio
        self.tiles_record = np.empty( (self.num_cycles, self.num_obs) )

        # TODO: print warning on minimum number of observations for generating tiles
        self.generate_tiles()

    def generate_tiles(self):
        # to have a more accurate fill ratio, generate correct proportion and then randomize order?

        ones = np.ones( int( np.round(self.fill_ratio * self.num_obs, 2) ) )
        zeros = np.zeros( int( np.round( (1 - self.fill_ratio) * self.num_obs , 2) ) )
        tiles = np.hstack( (ones, zeros) )

        assert(len(tiles) == self.num_obs)

        np.random.shuffle(tiles)

        return tiles

    def observe_color(self, tile_color, color_prob):
        """Provide observation of a tile color for one agent.
        """

        if (np.random.uniform() < color_prob):
            return tile_color
        else:
            return 1.0 - tile_color

    def compute_fisher_info(self, h, t, b, w):
        """Compute the Fisher information for one agent.
        """

        
        if h <= (1.0 - w) * t:
            return np.square(w) * np.square(w-1.0) / ( np.square(b+w-1.0) * (t*np.square(w) - 2*(t-h)*w + (t-h)) )
        elif h >= b*t:
            return np.square(b) * np.square(b-1.0) / ( np.square(b+w-1.0) * (t*np.square(b) - 2*h*b + h) )
        else:
            return h * (t-h) / ( np.power(t, 3) * np.square(b+w-1.0) )

    def _write_data_to_csv(self, f_hat_data, fisher_data, suffix=""):
        curr_time = datetime.now().strftime("%d%m%Y_%H%M%S")

        f_hat_filename = "./data/f_hat_" + curr_time + "_" + str(self.num_cycles) + "_" + str(self.num_obs) + "_" + suffix + ".csv"
        fisher_filename = "./data/fisher_" + curr_time + "_" + str(self.num_cycles) + "_" + str(self.num_obs) + "_" + suffix + ".csv"
        tiles_filename =  "./data/tiles_" + curr_time + "_" + str(self.num_cycles) + "_" + str(self.num_obs) + "_" + suffix + ".csv"

        with open(f_hat_filename, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(f_hat_data)

        with open(fisher_filename, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(fisher_data)

        with open(tiles_filename, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.tiles_record)

class SingleAgentSim(Sim):

    def __init__(self, num_cycles, num_obs, fill_ratio, b_prob, w_prob):

        super().__init__(num_cycles, num_obs, fill_ratio)

        self.b_prob = b_prob # P(black|black)
        self.w_prob = w_prob # P(white|white)
        self.agent_obs = np.zeros( (num_cycles, num_obs) )
        self.agent_avg_black_obs = np.zeros( (num_cycles, num_obs) )

        # Define data members
        self.f_hat = np.zeros( (num_cycles, num_obs) )
        self.fisher_info = np.zeros( (num_cycles, num_obs) )
        self.f_hat_sample_mean = np.zeros( num_obs )
        self.fisher_info_sample_mean = np.zeros( num_obs )
        self.f_hat_sample_std = np.zeros( num_obs )
        self.fisher_info_sample_std = np.zeros( num_obs )
        self.f_hat_sample_min = np.zeros( num_obs )
        self.fisher_info_sample_min = np.zeros( num_obs )
        self.f_hat_sample_max = np.zeros( num_obs )
        self.fisher_info_sample_max = np.zeros( num_obs )

    def run(self, data_flag = False):
        self.run_sim()

        self.compute_sample_mean()

        self.compute_sample_std()

        self.compute_sample_min()

        self.compute_sample_max()

        if data_flag: self.write_data()

    def compute_sample_mean(self):
        
        self.f_hat_sample_mean = np.mean(self.f_hat, axis=0)
        self.fisher_info_sample_mean = np.mean(self.fisher_info, axis=0)

    def compute_sample_std(self):

        # TODO: should i be using the biased or unbiased variance here? sticking with the unbiased for now
        self.f_hat_sample_std = np.std(self.f_hat, axis=0, ddof=1)
        self.fisher_info_sample_std = np.std(self.fisher_info, axis=0, ddof=1)

    def compute_sample_min(self):
        
        self.f_hat_sample_min = np.amin(self.f_hat, axis=0)
        self.fisher_info_sample_min = np.amin(self.fisher_info, axis=0)

    def compute_sample_max(self):
        
        self.f_hat_sample_max = np.amax(self.f_hat, axis=0)
        self.fisher_info_sample_max = np.amax(self.fisher_info, axis=0)

    def run_sim(self):

        # Compute the denominator term for the estimated fill ratio calculation
        denom = self.b_prob + self.w_prob - 1.0
        
        for cycle_ind in range(self.num_cycles):
            prev_obs = 0
            curr_obs = 0

            tiles = self.generate_tiles()

            for tile_ind, tile_color in enumerate(tiles):
                
                if tile_color == 1:
                    curr_obs = self.observe_color(tile_color, self.b_prob)
                else:
                    curr_obs = self.observe_color(tile_color, self.w_prob)

                # Store observations
                self.agent_obs[cycle_ind][tile_ind] = curr_obs
                self.agent_avg_black_obs[cycle_ind][tile_ind] = (prev_obs + curr_obs) / (tile_ind + 1) # tile number in the denom
                prev_obs += curr_obs

                # Compute estimated fill ratio
                if self.agent_avg_black_obs[cycle_ind][tile_ind] <= 1 - self.w_prob:
                    self.f_hat[cycle_ind][tile_ind] = 0.0
                elif self.agent_avg_black_obs[cycle_ind][tile_ind] >= self.b_prob:
                    self.f_hat[cycle_ind][tile_ind] = 1.0
                else:
                    self.f_hat[cycle_ind][tile_ind] = (self.agent_avg_black_obs[cycle_ind][tile_ind] + self.w_prob - 1.0) / denom

                # Compute Fisher information
                h = np.sum(self.agent_obs[cycle_ind][0:tile_ind+1])
                self.fisher_info[cycle_ind][tile_ind] = self.compute_fisher_info(h, tile_ind + 1, self.b_prob, self.w_prob)

            # Store the tile config
            self.tiles_record[cycle_ind] = tiles

    def write_data(self):
        suffix = "b" + str( int(self.b_prob*100) ) + "w" + str( int(self.w_prob*100) )

        self._write_data_to_csv( self.f_hat, self.fisher_info, suffix)