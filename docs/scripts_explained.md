# Utility scripts
This page describes scripts provided in this repository besides those that execute simulated experiments.

## Python Scripts (built in `collective_perception_static`)
The Python scripts are used for data processing, analysis, and visualization. To use them you will need `collective_perception_static` installed.

### Converting experiment data to visualization data
To visualize the experiment data, they have to be converted into visualization data. In addition to pure conversion, the outputs from multiple simulation executions can be combined into a single visualization data.

To visualize the data, a convergence threshold value $\delta$ must be provided; convergence is defined as the point in time when $|x^{k+1} - x^k| < \delta$ until the end of the experiment.

Depending on whether the output is from a static or a dynamic simulation, use `convert_*_to_viz_data_group.py` to combine and convert the data.
* Static: `convert_exp_data_to_viz_data_group.py <DIRECTORY-CONTAINING-ALL-SIMULATION-DATA> -s <CONVERTED-FILE-PATH>`
    ```
    usage: convert_exp_data_to_viz_data_group.py [-h] [-s S] FOLDER

    Load and convert ExperimentData objects into a VisualizationDataGroupStatic object.
    This script will convert and store ExperimentData objects with the same:
        - communication network type, and
        - target fill ratio range,
        - sensor probability range,
        - number of steps, and
        - number of trials,
    and with varying:
        - number of agents,
        - communication period,
        - communication probability.

    positional arguments:
      FOLDER      path to the top level directory containing the serialized ExperimentData files

    optional arguments:
      -h, --help  show this help message and exit
      -s S        path to store the pickled VisualizationDataGroupStatic object
    ```
* Dynamic: `convert_sim_stats_set_to_viz_data_group.py <DIRECTORY-CONTAINING-ALL-SIMULATION-DATA -s <CONVERTED-FILE-PATH>`
    ```
    usage: convert_sim_stats_set_to_viz_data_group.py [-h] [-s S] FOLDER

    Load and convert SimulationStatsSet protobuf files into a VisualizationDataGroupDynamic object.
    This script will convert and store SimulationStatsSet protobuf files with the same:
        - target fill ratio range,
        - sensor probability range,
        - number of steps, and
        - number of trials,
    and with varying:
        - robot speed, and
        - swarm density.

    positional arguments:
      FOLDER      path to the top level directory containing the serialized SimulationStatsSet protobuf files

    optional arguments:
      -h, --help  show this help message and exit
      -s S        path to store the pickled VisualizationDataGroupDynamic object
    ```
The conversion results in a new pickle file (containing a populated `VisualizationDataGroup` class instance).

>:warning: Depending on the size of the simulated experiment (i.e., simulation duration, number of agents, number of tested parameters tested in a single execution), the conversion process may take a substantial amount of time (10 minutes to 1 hour). It is recommended that you start small to gauge the required conversion time.

### Visualizing data
You can generate 3 forms of plots using the visualization data (i.e., the converted pickle file):

1. `series`: time-series plot.
2. `heatmap`: heatmap plot.
3. `scatter`: scatter plot.

Each of the 3 plot types is provided by a subcommand. Depending on whether the output is from a static or a dynamic simulation, use `visualize_multi_agent_data_*.py` to create the desired plot(s).
- Static: `visualize_multi_agent_data_static.py <OPTIONAL-FLAGS> <VISUALIZATION-DATA> <DESIRED-CONVERGENCE-THRESHOLD> <SUBCOMMANDS>`
    ```
    usage: visualize_multi_agent_data_static.py [-h] [-g] [-a] [-i] [-s] FILE CONV {series,heatmap,scatter} ...

    Visualize static multi-agent simulation data

    positional arguments:
      FILE                  path to folder containing serialized ExperimentData pickle files or path to a VisualizationDataGroup pickle file (see the "g" flag)
      CONV                  convergence threshold value
      {series,heatmap,scatter}
                              commands for visualization type
          series              visualize time series data
          heatmap             visualize heatmap data
          scatter             visualize scatter data

    optional arguments:
      -h, --help            show this help message and exit
      -g                    flag to indicate if the path is pointing to a VisualizationDataGroup pickle file
      -a                    flag to use aggregate data instead of data from individual trials (exclusive with the "-i" flag)
      -i                    flag to show individual agent data (only used for time series and scatter plot data with the "-U" flag; exclusive with the "-a" flag)
      -s                    flag to show the plots
    ```
- Dynamic: `visualize_multi_agent_data_dynamic.py <OPTIONAL-FLAGS> <VISUALIZATION-DATA> <DESIRED-CONVERGENCE-THRESHOLD> <SUBCOMMANDS>`
    ```
    usage: visualize_multi_agent_data_dynamic.py [-h] [-g] [-a] [-i] [-s] FILE CONV {series,heatmap,scatter} ...

    Visualize dynamic multi-agent simulation data

    positional arguments:
    FILE                  path to folder containing serialized SimulationStatsSet protobuf files or path to a VisualizationDataGroup pickle file (see the "g" flag)
    CONV                  convergence threshold value
    {series,heatmap,scatter}
                            commands for visualization type
        series              visualize time series data
        heatmap             visualize heatmap data
        scatter             visualize scatter data

    optional arguments:
      -h, --help            show this help message and exit
      -g                    flag to indicate if the path is pointing to a VisualizationDataGroup pickle file
      -a                    flag to use aggregate data instead of data from individual trials (exclusive with the "-i" flag)
      -i                    flag to show individual agent data (only used for time series and scatter plot data with the "-U" flag; exclusive with the "-a" flag)
      -s                    flag to show the plots
    ```

Script help for the subcommands:
- `series` general usage:
    ```
    usage: visualize_multi_agent_data_*.py FILE CONV series [-h] -TFR TFR -SP SP -U [U [U ...]]

    optional arguments:
      -h, --help      show this help message and exit
      -TFR TFR        single target fill ratio to use in plotting time series data
      -SP SP          single sensor probability to use in plotting time series data
      <SPECIFIC-HELP-SEE-BELOW>
    ```
    with the static version:
    ```
      -U [U [U ...]]  communications period, communication probability, and number of agents to use in plotting time series data
    ```
    and the dynamic version:
    ```
      -U [U [U ...]]  robot speed and swarm density to use in plotting time series data
    ```
- `heatmap` general usage:
    ```
    usage: visualize_multi_agent_data_*.py FILE CONV heatmap [-h] [-u [U [U ...]]] [-rstr RSTR [RSTR ...]] [-row ROW [ROW ...]] [-cstr CSTR [CSTR ...]] [-col COL [COL ...]]

    optional arguments:
      -h, --help            show this help message and exit
      <SPECIFIC-HELP-SEE-BELOW>
      -rstr RSTR [RSTR ...]
                              (optional) outer grid row labels (must match number of "ROW" arguments; unused if "-u" arguments are used)
      -row ROW [ROW ...]    (optional) outer grid row coordinates (unused if "-u" arguments are used)
      -cstr CSTR [CSTR ...]
                              (optional) outer grid column labels (must match number of "COL" arguments; unused if "-u" arguments are used)
      -col COL [COL ...]    (optional) outer grid column coordinates (unused if "-u" arguments are used)
    ```
    with the static version:
    ```
      -u [U [U ...]]        (optional) communications period, communication probability, and number of agents to use in plotting single heatmap data
    ```
    and the dynamic version:
    ```
      -u [U [U ...]]        (optional) robot speed and swarm density to use in plotting single heatmap data
    ```
- `scatter` general usage:
    ```
    usage: visualize_multi_agent_data_*.py FILE CONV scatter [-h] [-tfr TFR [TFR ...]] [-sp SP [SP ...]] -U [U [U ...]]

    optional arguments:
      -h, --help          show this help message and exit
      -tfr TFR [TFR ...]  (optional) target fill ratio to use in plotting scatter data (must provide single "-sp" argument if this is not provided)
      -sp SP [SP ...]     (optional) sensor probability to use in plotting scatter data (must provide single "-tfr" argument if this is not provided)
      <SPECIFIC-HELP-SEE-BELOW>
    ```
    with the static version:
    ```
      -U [U [U ...]]      communications period, communication probability, and number of agents to use in plotting scatter data
    ```
    and the dynamic version:
    ```
      -U [U [U ...]]      robot speed and swarm density to use in plotting scatter data
    ```

### Obtaining serialized data information
`serial_data_info.py` displays information about a serialized pickle file. (TODO: currently only supports static simulation experiment outputs, i.e., pickled `ExperimentData` files):
```
$ serial_data_info.py <FILE>
```

Script help:
```
usage: serial_data_info.py [-h] DATA

Display information of the serialized data.

positional arguments:
  DATA        Filename to the data file.
```

## C++ scripts (built in `collective_perception_dynamic`)

### Computing wall positions corresponding to a desired swarm density
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

### Obtaining serialized data information
`protobuf_info` displays information about a serialized protobuf file:
```
$ protobuf_info <FILE>
```

## Bash
### Running batch simulated experiment executions
As described in the [README](../README.md), a single simulation execution is defined to have a fixed values for the outer paramater group and varying values for the inner parameter group. The two `hpc_execute_multi_agent_sim_*.sh` bash scripts provided in `scripts/bash/` go one level further, letting you run multiple executions. *The scripts will run using the containerized simulator only, but modifying them to run a local build should be simple.*

For example, you can choose a range of robot speeds and swarm densities (outer parameter group) for a dynamic topology simulation. Simply add the desired `SPEED` and `POSITION` values in the bash array:
```bash
# Define varying parameters
SPEED=(10.0 15.0 20.0 25.0) # cm/s
POSITION=(4.436599480604251 3.1517942390846527 2.011746925739275 1.4371645541621039) # for density = (1 2 5 10) with number of agents = 50
```
Then execute with the appropriate arguments
```
$ ./hpc_execute_multi_agent_sim_dynamic.sh param_multi_agent_sim_dynamic.argos multi_agent_sim_full_no_qt.sif output_data
```

## Slurm
A number of scripts are provided to run batch jobs using the SLURM batch manager on a HPC cluster in `scripts/slurm/`. The `sbatch_multi_agent_sim_*.sh` scripts use the `hpc_execute_multi_agent_sim_*.sh` scripts under the hood, so you will need to adjust the experimental parameters accordingly.

## Apptainer definition files
The definition files to build the Apptainers (containers) are provided in `apptainer/def/`. See the [official Apptainer documentation](https://apptainer.org/docs/) for more details.

## MATLAB scripts (no longer maintained)
The MATLAB scripts are mostly for quick prototyping and data processing/visualization, and is no longer maintained.

### Running single robot simulated experiment
To simulate a single agent for **one target fill ratio and one sensor probability**, create/modify a `param_simple_single_agent_no_motion.m` file to set the desired parameters in the same folder, then run the `simple_single_agent_no_motion.m` script in MATLAB.

The parameter file should look like the following:
```matlab
%% param_simple_single_agent_no_motion.m

N = <INT>;                      % total number of observations
b = <FLOAT>;                    % sensor probability to black tile
w = b;                          % sensor probability to white tile
sim_cycles = <INT>;             % number of agents to simulate (or simulation cycles for one agent)
desired_fill_ratio = <FLOAT>;   % fill ratio, f (can be set to rand(1))
```
### Visualizing data
Use `plot_mean_rel_err.m` for plotting the averaged `f_hat` errors for multiple simulations. *Currently only for `c200_o100`, `c200_o250`, `c200_o500`, and `c200_o1000` files.*

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