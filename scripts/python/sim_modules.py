import numpy as np
import csv
import os
from datetime import datetime

class Sim:

    def __init__(self, num_cycles, num_obs, des_fill_ratio):
        self.num_cycles = num_cycles
        self.num_obs = num_obs
        self.des_fill_ratio = des_fill_ratio
        self.avg_fill_ratio = 0.0
        self.tiles_record = np.empty( (self.num_cycles, self.num_obs) )

        # TODO: print warning on minimum number of observations for generating tiles
        self.generate_tiles()

    def generate_tiles(self):
        """Generate the tiles based on the desired/nominal fill ratios.
        """
        
        # Draw bernoulli samples for tiles based on desired fill ratio
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

    def compute_fisher_inv(self, h, t, b, w):
        """Compute the Fisher information for one agent.
        """

        
        if h <= (1.0 - w) * t:
            return np.square(w) * np.square(w-1.0) / ( np.square(b+w-1.0) * (t*np.square(w) - 2*(t-h)*w + (t-h)) )
        elif h >= b*t:
            return np.square(b) * np.square(b-1.0) / ( np.square(b+w-1.0) * (t*np.square(b) - 2*h*b + h) )
        else:
            return h * (t-h) / ( np.power(t, 3) * np.square(b+w-1.0) )

    def _write_data_to_csv(self, f_hat_data, fisher_inv_data, suffix=""):
        curr_time = datetime.now().strftime("%m%d%Y_%H%M%S")

        f_hat_filename = "./data/f_hat_" + curr_time + "_" + str(self.num_cycles) + "_" + str(self.num_obs) + "_" + suffix + ".csv"
        fisher_inv_filename = "./data/fisher_inv_" + curr_time + "_" + str(self.num_cycles) + "_" + str(self.num_obs) + "_" + suffix + ".csv"
        tiles_filename =  "./data/tiles_" + curr_time + "_" + str(self.num_cycles) + "_" + str(self.num_obs) + "_" + suffix + ".csv"

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

    def __init__(self, num_cycles, num_obs, des_fill_ratio, b_prob, w_prob):

        super().__init__(num_cycles, num_obs, des_fill_ratio)

        self.b_prob = b_prob # P(black|black)
        self.w_prob = w_prob # P(white|white)
        self.agent_obs = np.zeros( (num_cycles, num_obs) )
        self.agent_avg_black_obs = np.zeros( (num_cycles, num_obs) )

        # Define data members
        self.f_hat = np.zeros( (num_cycles, num_obs) )
        self.fisher_inv = np.zeros( (num_cycles, num_obs) )
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

        if data_flag: self.write_data()

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
                self.fisher_inv[cycle_ind][tile_ind] = self.compute_fisher_inv(h, tile_ind + 1, self.b_prob, self.w_prob)

            # Store the tile config
            self.tiles_record[cycle_ind] = tiles

            # Compute the average tile ratio up to this siimulation cycle
            self.avg_fill_ratio =  (cycle_ind)/(cycle_ind+1) * self.avg_fill_ratio + 1/(cycle_ind+1) * sum(tiles)/self.num_obs

    def write_data_to_csv(self):
        suffix = "b" + str( int(self.b_prob*100) ) + "w" + str( int(self.w_prob*100) )

        self._write_data_to_csv( self.f_hat, self.fisher_inv, suffix )

class HeatmapData:
    """Class to store and process heatmap data
    """

    def __init__(self, num_cycle, num_obs, sensor_prob_range, fill_ratio_range):

        self.num_cycle = num_cycle
        self.num_obs = num_obs
        self.sensor_prob_range = sensor_prob_range
        self.fill_ratio_range = fill_ratio_range

        self.sensor_prob_inc = self.sensor_prob_range[1] - self.sensor_prob_range[0]
        self.fill_ratio_inc = self.fill_ratio_range[1] - self.fill_ratio_range[0]

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
    
        min_sensor_prob, max_sensor_prob = int(self.sensor_prob_range[0]*1e2), int(self.sensor_prob_range[-1]*1e2)
        min_des_fill_ratio, max_des_fill_ratio = int(self.fill_ratio_range[0]*1e2), int(self.fill_ratio_range[-1]*1e2)        
        sensor_prob_inc, fill_ratio_inc = int(self.sensor_prob_inc*1e2), int(self.fill_ratio_inc*1e2)
        
        prob_suffix = "_p" + str(min_sensor_prob) + "-" + str(sensor_prob_inc) + "-" + str(max_sensor_prob)
        f_suffix = "_f" + str(min_des_fill_ratio) + "-" + str(fill_ratio_inc) + "-" + str(max_des_fill_ratio)
        suffix = prob_suffix + f_suffix
        
        # Write heatmap data
        f_hat_mean_filename = "f_hat_heatmap_mean_" + str(self.num_cycle) + "_" + str(self.num_obs) + suffix + ".csv"
        f_hat_min_filename = "f_hat_heatmap_min_" + str(self.num_cycle) + "_" + str(self.num_obs) + suffix + ".csv"    
        f_hat_max_filename = "f_hat_heatmap_max_" + str(self.num_cycle) + "_" + str(self.num_obs) + suffix + ".csv"
        fisher_inv_mean_filename = "fisher_inv_heatmap_mean_" + str(self.num_cycle) + "_" + str(self.num_obs) + suffix + ".csv"
        fisher_inv_min_filename = "fisher_inv_heatmap_min_" + str(self.num_cycle) + "_" + str(self.num_obs) + suffix + ".csv"    
        fisher_inv_max_filename = "fisher_inv_heatmap_max_" + str(self.num_cycle) + "_" + str(self.num_obs) + suffix + ".csv"
        des_f_avg_f_filename = "des_f_avg_f_" + str(self.num_cycle) + "_" + str(self.num_obs) + suffix + ".csv"
        

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

    def populate(self, single_agent_sim_obj):
        """Populate data given SingleAgentSim object.
        """

        self.f_hat_mean.append( single_agent_sim_obj.f_hat_sample_mean[-1] )
        self.f_hat_min.append( single_agent_sim_obj.f_hat_sample_min[-1] )
        self.f_hat_max.append( single_agent_sim_obj.f_hat_sample_max[-1] )

        self.fisher_inv_mean.append( single_agent_sim_obj.fisher_inv_sample_mean[-1] )
        self.fisher_inv_min.append( single_agent_sim_obj.fisher_inv_sample_min[-1] )
        self.fisher_inv_max.append( single_agent_sim_obj.fisher_inv_sample_max[-1] )
        
        self.avg_f = single_agent_sim_obj.avg_fill_ratio