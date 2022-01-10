# Collective Consensus Project

## MATLAB Scripts
The MATLAB scripts are mostly for quick prototyping and data processing/visualization.

### `plot_heat_surf.m`
Used for plotting heatmap data from the following files (obtained from running the sing):
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

## Python Scripts
The Python scripts are used for major simulation runs (simulations that can generate heatmap data).

### `single_agent_sim.py`
Used for running multiple simulations; simulates a single agent (with no communication) across **multiple sensor probabilities and desired fill ratios**.

To run the script, create/modify a `param_single_agent_sim.yaml` file to set the desired simulation parameters in the same folder, then run
```
$ python3 single_agent_sim.py
```
which would produce CSV files that can be used for further analysis.

The parameter file should look like the following:
```yaml
# param_single_agent_sim.yaml

--- # document start marker
sensorProb:
  min: <FLOAT> # min sensor probability
  max: <FLOAT> # max sensor probability
  incSteps: <INT> # number of steps from min to max inclusive
desFillRatios:
  min: <FLOAT> # min desired fill ratio
  max: <FLOAT> # max desired fill ratio
  incSteps: <INT> # number of steps from min to max inclusive
numCycle: <INT> # number of cycles to run
numObs: <INT> # number of observations per cycle/agent
writeAllData: <BOOL> # whether to write data from every single experiment
... # document end marker
```

Note: