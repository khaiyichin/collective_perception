%% Parameter file for plot_heat_surf.m

parent_folder = "C:\Users\khaiy\Documents\collective_consensus\scripts\python\data\notable_data\";

fisher_inv_filepath = parent_folder + "01072022_123813_c200_o1000_p55-2-100_f5-5-95\fisher_inv_heatmap_mean_c200_o1000_p55-2-100_f5-5-95.csv";
f_hat_filepath = parent_folder + "01072022_123813_c200_o1000_p55-2-100_f5-5-95\f_hat_heatmap_mean_c200_o1000_p55-2-100_f5-5-95.csv";
sim_fill_ratios_filepath = parent_folder + "01072022_123813_c200_o1000_p55-2-100_f5-5-95\des_f_avg_f_c200_o1000_p55-2-100_f5-5-95.csv";

xrange = 0.55:0.025:1.0; % sensor prob
yrange = 0.05:0.05:0.95; % fill ratios
sim_cycles = 200;
num_obs = 1000;