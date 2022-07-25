# Utility scripts
In addition to the scripts that execute simulated experiments, scripts to process, analyze, and visualize data are provided; this page describes all of them.

## Python Scripts (built in `collective_perception_static`)
The Python scripts are used for data processing, analysis, and visualization.

  <!-- - Describe modules:
    - how to use the classes, what do the classes do and where do they fit?
  - Describe python scripts:
    - how can they be used?
    - what arguments are needed? -->

### `convert_exp_data_to_viz_data_group.py`

### `convert_sim_stats_set_to_viz_data_group.py`

### `serial_data_info.py`

### `visualize_multi_agent_data_static.py`

### `visualize_multi_agent_data_dynamic.py`

## C++ scripts (built in `collective_perception_dynamic`)

### `compute_wall_positions.cpp`
To test a specific swarm density using the dynamic topology simulator, you need to specify the appropriate wall position to constrain the robots' walkable area, as the following equation describes:

$$ D = \frac{\text{communication area}}{\text{walkable area}} = \frac{N \pi r^2}{L^2} $$

Simply run the `compute_wall_positions.cpp` script and answer the prompts accordingly:

```
$ compute_wall_positions

Please specify the desired number of robots: 25
Please specify the desired swarm density: 10
Please specify the radius of the communication range of the robot in m: 0.7
Please specify the desired wall thickness in m (optional): 0.1
Please specify the desired wall height in m (optional): 0.5
```
Upon completion the script will generate an XML output that you can copy and paste in your `.argos` configuration file.
```
Configuration for the .argos file (copy and paste this under the <arena> node):

	<box id="wall_north" size="3.000000, 0.100000, 0.500000" movable="false">
	    <body position="0.000000, 1.030873, 0.000000" orientation="0, 0, 0" />
	</box>
	<box id="wall_south" size="3.000000, 0.100000, 0.500000" movable="false">
	    <body position="0.000000, -1.030873, 0.000000" orientation="0, 0, 0" />
	</box>
	<box id="wall_east" size="0.100000, 3.000000, 0.500000" movable="false">
	    <body position="1.030873, 0.000000, 0.000000" orientation="0, 0, 0" />
	</box>
	<box id="wall_west" size="0.100000, 3.000000, 0.500000" movable="false">
	    <body position="-1.030873, 0.000000, 0.000000" orientation="0, 0, 0" />
	</box>

	<distribute>
	    <position method="uniform" min="-1.030873, -1.030873, 0" max="1.030873, 1.030873, 0" />
	    <orientation method="uniform" min="0, 0, 0" max="0, 0, 0" />
	    <entity quantity="25" max_trials="100" base_num="0">
	        <kheperaiv id="kiv" rab_data_size="50" rab_range="0.700000">
	            <controller config="bck" />
	        </kheperaiv>
	    </entity>
	</distribute>
```

## Bash
- `bash`: scripts to execute multi-parameter simulations.
  <!-- - Describe bash scripts:
    - how can they be used? -->

## Slurm
- `slurm`: scripts to run jobs using the SLURM batch manager on a HPC cluster.
  <!-- - how to use the slurm scripts?
    - what arguments are required?
    - where do you run them from? -->


## Apptainer definition files
<!-- - what is the def file for?
  - how does it work with the bash scripts to run hpc simulation? -->

## MATLAB scripts
The MATLAB scripts are mostly for quick prototyping and data processing/visualization.

### `plot_heat_surf.m`
Used for plotting heatmap data from the following files (obtained from running `single_agent_sim.py`):
- `fisher_inv_heatmap_*.csv`,
- `f_hat_heatmap_*.csv`,
- `des_f_avg_f_*.csv`.
The script outputs heatmap and 3-D surface plots for `f_hat` and `fisher_inv`.

To run the script, create/modify a `param_plot_heat_surf.m` file to set the desired parameters in the same folder, then run the `plot_heat_surf.m` script in MATLAB.

The parameter file should look like the following:
```matlab
%% param_plot_heat_surf.m

fisher_inv_filepath = <STRING>;       % path to Fisher inverse heatmap data file
f_hat_filepath = <STRING>;            % path to f_hat heatmap data file
sim_fill_ratios_filepath = <STRING>;  % path to fill ratio file

xrange = <FLOAT ARRAY>;               % range of sensor probabilities
yrange = <FLOAT ARRAY>;               % range of fill ratios
sim_cycles = <INT>;                   % number of agents to simulate (or simulation cycles for one agent)
num_obs = <INT>;                      % total number of observations
```

### `plot_mean_rel_err.m`
Used for plotting the averaged `f_hat` errors for multiple simulations. *Currently only for `c200_o100`, `c200_o250`, `c200_o500`, and `c200_o1000` files.*

To run the script, create/modify a `param_plot_mean_rel_err.m` file to set the desired parameters in the same folder, then run the `plot_heat_surf.m` script in MATLAB.

The parameter file should look like the following:
```matlab
%% param_plot_mean_rel_err.m

c200_o100_filepaths = {
    [ <STRING> ; <STRING> ];  % path to heatmap data file ; path to fill ratio file
    [ <STRING> ; <STRING> ];  % path to heatmap data file ; path to fill ratio file
    [ <STRING> ; <STRING> ];  % path to heatmap data file ; path to fill ratio file
    ...
};

c200_o250_filepaths = {
    [ <STRING> ; <STRING> ];  % path to heatmap data file ; path to fill ratio file
    [ <STRING> ; <STRING> ];  % path to heatmap data file ; path to fill ratio file
    [ <STRING> ; <STRING> ];  % path to heatmap data file ; path to fill ratio file
    ...
};

c200_o500_filepaths = {
    [ <STRING> ; <STRING> ];  % path to heatmap data file ; path to fill ratio file
    [ <STRING> ; <STRING> ];  % path to heatmap data file ; path to fill ratio file
    [ <STRING> ; <STRING> ];  % path to heatmap data file ; path to fill ratio file
    ...
};

c200_o1000_filepaths = {
    [ <STRING> ; <STRING> ];  % path to heatmap data file ; path to fill ratio file
    [ <STRING> ; <STRING> ];  % path to heatmap data file ; path to fill ratio file
    [ <STRING> ; <STRING> ];  % path to heatmap data file ; path to fill ratio file
    ...
};

xrange = <FLOAT ARRAY>;       % range of sensor probabilities
yrange = <FLOAT ARRAY>;       % range of fill ratios
sim_cycles = <INT>;           % number of agents to simulate (or simulation cycles for one agent)
num_obs = <INT>;              % total number of observations
```

### `simple_single_agent_no_motion.m`
Used for running a quick simulation/prototyping; simulates a single agent (with no communication) for **one desired fill ratio and one sensor probability**.

To run the script, create/modify a `param_simple_single_agent_no_motion.m` file to set the desired parameters in the same folder, then run the `simple_single_agent_no_motion.m` script in MATLAB.

The parameter file should look like the following:
```matlab
%% param_simple_single_agent_no_motion.m

N = <INT>;                      % total number of observations
b = <FLOAT>;                    % sensor probability to black tile
w = b;                          % sensor probability to white tile
sim_cycles = <INT>;             % number of agents to simulate (or simulation cycles for one agent)
desired_fill_ratio = <FLOAT>;   % fill ratio, f (can be set to rand(1))
```