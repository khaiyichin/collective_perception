%% Parameter file for plot_mean_rel_err.m

parent_folder = "C:\Users\khaiy\Documents\collective_consensus\scripts\python\data\notable_data\";

% Data file paths in the following format:
% param_1_filepaths = 
%   {
%       [
%           param_1_heatmap_data_1; param_1_fill_ratio_data_1
%       ];
%       [
%           param_1_heatmap_data_2; param_1_fill_ratio_data_2
%       ];
%       [
%           param_1_heatmap_data_3; param_1_fill_ratio_data_3
%       ];
%       ...
%   };
% 
% param_2_filepaths = 
%   {
%       [
%           param_2_heatmap_data_1; param_2_fill_ratio_data_1
%       ];
%       [
%           param_2_heatmap_data_2; param_2_fill_ratio_data_2
%       ];
%       [
%           param_2_heatmap_data_3; param_2_fill_ratio_data_3
%       ];
%       ...
%   };

c200_o100_filepaths = {
    [
        parent_folder + "01052022_170435_c200_o100_p55-2-100_f5-5-95\f_hat_heatmap_mean_c200_o100_p55-2-100_f5-5-95.csv";
        parent_folder + "01052022_170435_c200_o100_p55-2-100_f5-5-95\des_f_avg_f_c200_o100_p55-2-100_f5-5-95.csv"
    ];
    [
        parent_folder + "01052022_172932_c200_o100_p55-2-100_f5-5-95\f_hat_heatmap_mean_c200_o100_p55-2-100_f5-5-95.csv";
        parent_folder + "01052022_172932_c200_o100_p55-2-100_f5-5-95\des_f_avg_f_c200_o100_p55-2-100_f5-5-95.csv"
    ];
    [
        parent_folder + "01062022_095829_c200_o100_p55-2-100_f5-5-95\f_hat_heatmap_mean_c200_o100_p55-2-100_f5-5-95.csv";
        parent_folder + "01062022_095829_c200_o100_p55-2-100_f5-5-95\des_f_avg_f_c200_o100_p55-2-100_f5-5-95.csv"
    ]
};

c200_o250_filepaths = {
    [
        parent_folder + "01052022_164359_c200_o250_p55-2-100_f5-5-95\f_hat_heatmap_mean_c200_o250_p55-2-100_f5-5-95.csv";
        parent_folder + "01052022_164359_c200_o250_p55-2-100_f5-5-95\des_f_avg_f_c200_o250_p55-2-100_f5-5-95.csv"
    ];
    [
        parent_folder + "01052022_165332_c200_o250_p55-2-100_f5-5-95\f_hat_heatmap_mean_c200_o250_p55-2-100_f5-5-95.csv";
        parent_folder + "01052022_165332_c200_o250_p55-2-100_f5-5-95\des_f_avg_f_c200_o250_p55-2-100_f5-5-95.csv"
    ];
    [
        parent_folder + "01062022_153731_c200_o250_p55-2-100_f5-5-95\f_hat_heatmap_mean_c200_o250_p55-2-100_f5-5-95.csv";
        parent_folder + "01062022_153731_c200_o250_p55-2-100_f5-5-95\des_f_avg_f_c200_o250_p55-2-100_f5-5-95.csv"
    ]
};

c200_o500_filepaths = {
    [
        parent_folder + "01072022_122101_c200_o500_p55-2-100_f5-5-95\f_hat_heatmap_mean_c200_o500_p55-2-100_f5-5-95.csv";
        parent_folder + "01072022_122101_c200_o500_p55-2-100_f5-5-95\des_f_avg_f_c200_o500_p55-2-100_f5-5-95.csv"
    ];
    [
        parent_folder + "12232021_141120_c200_o500_p55-2-100_f5-5-95\f_hat_heatmap_mean_c200_o500_p55-2-100_f5-5-95.csv";
        parent_folder + "12232021_141120_c200_o500_p55-2-100_f5-5-95\des_f_avg_f_c200_o500_p55-2-100_f5-5-95.csv"
    ];
    [
        parent_folder + "12232021_154103_c200_o500_p55-2-100_f5-5-95\f_hat_heatmap_mean_c200_o500_p55-2-100_f5-5-95.csv";
        parent_folder + "12232021_154103_c200_o500_p55-2-100_f5-5-95\des_f_avg_f_c200_o500_p55-2-100_f5-5-95.csv"
    ]
};

c200_o1000_filepaths = {
    [
        parent_folder + "12232021_120428_c200_o1000_p55-2-100_f5-5-95\f_hat_heatmap_mean_c200_o1000_p55-2-100_f5-5-95.csv";
        parent_folder + "12232021_120428_c200_o1000_p55-2-100_f5-5-95\des_f_avg_f_c200_o1000_p55-2-100_f5-5-95.csv"
    ];
    [
        parent_folder + "12232021_145704_c200_o1000_p55-2-100_f5-5-95\f_hat_heatmap_mean_c200_o1000_p55-2-100_f5-5-95.csv";
        parent_folder + "12232021_145704_c200_o1000_p55-2-100_f5-5-95\des_f_avg_f_c200_o1000_p55-2-100_f5-5-95.csv"
    ];
    [
        parent_folder + "01072022_123813_c200_o1000_p55-2-100_f5-5-95\f_hat_heatmap_mean_c200_o1000_p55-2-100_f5-5-95.csv";
        parent_folder + "01072022_123813_c200_o1000_p55-2-100_f5-5-95\des_f_avg_f_c200_o1000_p55-2-100_f5-5-95.csv"
    ]
};

xrange = 0.55:0.025:1.0; % sensor prob
yrange = 0.05:0.05:0.95; % fill ratios
sim_cycles = 200;
num_obs = 100;