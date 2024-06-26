# Utility scripts
This page describes scripts provided in this repository besides those that execute simulated experiments. *For information on benchmark algorithms (how to run, implement, and analyze), see [here](./benchmark_algo_explained.md).*

## Python Scripts (built in `collective_perception_static`)
The Python scripts are used for data processing, analysis, and visualization. To use them you will need `collective_perception_static` installed.

### Converting experiment data to visualization data
To visualize the experiment data, they have to be converted into visualization data. In addition to pure conversion, the outputs from multiple simulation executions can be combined into a single visualization data.

To plot the visualization data, a convergence threshold value $\delta$ must be provided; convergence is defined as the point in time when $|x^{k+1} - x^k| < \delta$ until the end of the experiment.

Depending on whether the output is from a static or a dynamic simulation, use `convert_*_to_viz_data_group.py` to combine and convert the data.
* Static:
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
* Dynamic:
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
The conversion results in a new `.vdg` pickle file (containing a populated `VisualizationDataGroup` class instance). You can then use this file to [visualize the data](#visualizing-data).

<details><summary>Example: converting static experiment data</summary>

Given a directory `data` containing the following static simulation data files:
  - `agt100_prd10_prob1.ped` = 100 agents, comms. period of 1,
  - `agt1000_prd10_prob1.ped` = 1000 agents, comms. period of 10,
  - `agt100_prd1_prob1.ped` = 100 agents, comms. period of 1, and
  - `agt1000_prd1_prob1.ped` = 1000 agents, comms. period of 1,

they can be combined as long as their
  - communication network type,
  - target fill ratio *range*,
  - sensor probability *range*,
  - number of steps,
  - and number of trials,

are the same. For instance, they could all have the following experimental parameters:
  - network type = `scale-free`,
  - target fill ratios = `{0.55, 0.95}`
  - sensor probability range = `[0.525, 0.975]`
  - number of steps = `10000`
  - number of trials = `30`

Then the conversion can happen as follows.
```
$ convert_exp_data_to_viz_data_group.py data -s /home/user/converted_from_ped.vdg
Saved VisualizationDataGroupStatic object containing 4 items at: /home/usr/converted_from_ped.vdg.
```

</details>

<details><summary> Example: converting dynamic experiment data </summary>

Given a directory `data` containing dynamic simulation data files:
  - `den1_spd10.pbs` = swarm density of 1, robot speed of 10 cm/s,
  - `den10_spd10.pbs` = swarm density of 10, robot speed of 10 cm/s,

they can be combined as long as their
  - target fill ratio *range*,
  - sensor probability *range*,
  - number of steps,
  - and number of trials,

are the same. For instance, they could all have the following experimental parameters:
  - target fill ratios = `{0.55, 0.95}`
  - sensor probability range = `[0.525, 0.975]`
  - number of steps = `10000`
  - number of trials = `30`

Then the conversion can happen as follows.
```
$ convert_sim_stats_to_viz_data_group.py data -s /home/user/converted_from_pbs.vdg
Saved VisualizationDataGroupDynamic object containing 2 items at: /home/user/converted_from_pbs.vdg.
```

</details>

>:warning: Depending on the size of the simulated experiment (i.e., simulation duration, number of agents, number of tested parameters tested in a single execution), the conversion process may take a substantial amount of time (10 minutes to 1 hour). It is recommended that you start small to gauge the required conversion time.

### Visualizing data
>:info: This subsection enumerates some visualization options that have been created for you to use. If you want to manipulate the data directly and create your own figures, go to the [Process Data Manually](#process-data-manually) subsection below.

You can generate 3 forms of plots using the visualization data (i.e., the converted pickle file):

1. `series`: time-series plot of agent estimates, for a fixed inner parameter pair (target fill ratio and sensor probability) and a fixed outer parameter pair.
2. `heatmap`: heatmap plot, containing average agent estimate values across for all the inner and outer parameters.
3. `scatter`: scatter plot of agents' estimates across all trials for a fixed outer parameter pair and one fixed inner parameter (either target fill ratio or sensor probability).
4. `decision`: collective decision plot of agents' decisions (through bins picked based on their informed estimates).

**Each of the 4 plot types is provided by a subcommand (required argument)**. Depending on whether the output is from a static or a dynamic simulation, use `visualize_multi_agent_data_*.py` to create the desired plot(s).

The static and dynamic visualization scripts are only different when it comes to subcommand usage. Both have the same help message for the main command (with (*) denoting `static`/`dynamic` and (**) denoting `ExperimentData pickle`/`SimulationStatsSet protobuf`):

```
usage: visualize_multi_agent_data_(*).py [-h] [-g] [-a] [-i] [-s] [--steps STEPS] FILE CONV {series,heatmap,scatter,decision} ...

Visualize (*) multi-agent simulation data

positional arguments:
  FILE                  path to folder containing serialized (**) pickle files or path to a VisualizationDataGroup pickle file (see the "g" flag)
  CONV                  convergence threshold value
  {series,heatmap,scatter,decision}
                        commands for visualization type
    series              visualize time series data
    heatmap             visualize heatmap data
    scatter             visualize scatter data
    decision            visualize collective-decision making data

optional arguments:
  -h, --help            show this help message and exit
  -g                    flag to indicate if the path is pointing to a VisualizationDataGroup pickle file
  -a                    flag to use aggregate data instead of data from individual trials (exclusive with the "-i" flag)
  -i                    flag to show individual agent data (only used for time series and scatter plot data with the "-U" flag; exclusive with the "-a" flag)
  -s                    flag to show the plots
  --steps STEPS         first n simulation steps to evaluate to (default: evaluate from start to end of simulation)
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
    with specific help for the static version:
    ```
      -U [U [U ...]]  communications period, communication probability, and number of agents to use in plotting time series data
    ```
    and the dynamic version:
    ```
      -U [U [U ...]]  robot speed and swarm density to use in plotting time series data
    ```

    <details><summary>Example: visualize time series plot</summary>

    Suppose we are interested in seeing the time-series behavior of the robots in the dynamic simulation with a convergence threshold of 0.01.

    Specifically, we are interested in seeing the experiment with the following parameters:
    - homogeneous robots of `0.525` sensor probability,
    - environment fill ratio of `0.05`,
    - robot speed of `10` cm/s, and
    - swarm density of `1`.

    Then the command is as follows:
    ```
    $ visualize_multi_agent_data_dynamic.py /home/user/converted_from_pbs.vdg 0.01 -gsi series -TFR 0.05 -SP 0.525 -U 10 1
    ```
    This will only create figures for viewing without saving (you have to save them manually).

    </details>

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
    with specific help for the static version:
    ```
      -u [U [U ...]]        (optional) communications period, communication probability, and number of agents to use in plotting single heatmap data
    ```
    and the dynamic version:
    ```
      -u [U [U ...]]        (optional) robot speed and swarm density to use in plotting single heatmap data
    ```

    <details><summary>Example: visualize heatmap</summary>

    Suppose we are interested in seeing the heatmap of the robot estimates in the static simulation with a convergence threshold of 0.01.

    Specifically, we are interested in seeing the experiment with the following parameters:
    - homogeneous robots with sensor probability ranging from `0.525` to `0.975`,
    - environment fill ratio ranging from `0.05` to `0.95`,
    - scale-free network,
    - communication period ranging from `1` to `10`, and
    - number of agents ranging from `10` to `100`.

    Then the command is as follows (needs testing since repository update):
    ```
    $ visualize_multi_agent_data_static.py /home/user/converted_from_sf_network_ped.vdg 0.01 -gs heatmap -rstr "Num = 10" "Num = 20" "Num = 50" "Num = 100" -row 10 20 50 100 -cstr "Period = 1" "Period = 2" "Period = 5" "Period = 10" -col 1 2 5 10
    ```

    <img src="./img/heatmap_sta_conv10_s2000_t5.png" width="1000" />

    </details>

- `scatter` general usage:
    ```
    usage: visualize_multi_agent_data_*.py FILE CONV scatter [-h] [-tfr TFR [TFR ...]] [-sp SP [SP ...]] -U [U [U ...]]

    optional arguments:
      -h, --help          show this help message and exit
      -tfr TFR [TFR ...]  (optional) target fill ratio to use in plotting scatter data (must provide single "-sp" argument if this is not provided)
      -sp SP [SP ...]     (optional) sensor probability to use in plotting scatter data (must provide single "-tfr" argument if this is not provided)
      <SPECIFIC-HELP-SEE-BELOW>
    ```
    with specific help for the static version:
    ```
      -U [U [U ...]]      communications period, communication probability, and number of agents to use in plotting scatter data
    ```
    and the dynamic version:
    ```
      -U [U [U ...]]      robot speed and swarm density to use in plotting scatter data
    ```

    <details><summary>Example: visualize scatter plot</summary>

    Suppose we are interested in seeing the scatter plot of robot estimates in the static simulation with a convergence threshold of 0.01.

    Specifically, we are interested in seeing the experiment with the following parameters:
    - homogeneous robots with sensor probabilities `{0.525, 0.575, 0.625, ..., 0.975}`, and heterogeneous robots with uniformly distributed sensor probabilities `U(0.525, 0.975)`,
    - environment fill ratio of `0.75`,
    - line network,
    - communication period of `10`, and
    - number of agents of `100`.

    Then the command is as follows:
    ```
    $ visualize_multi_agent_data_static.py /home/user/converted_from_line_network_ped.vdg 0.01 -gsi scatter -tfr 0.75 -U 10 1 100
    ```

    <img src="./img/scatter_sta_conv10_s10000_t30_tfr750_prd10_cprob1_agt100.png" width="1000" />

    </details>

- `decision` general usage:
    ```
    usage: visualize_multi_agent_data_dynamic.py FILE CONV decision [-h] -TFR TFR [-sp SP [SP ...]] -U U [U ...] [--bins BINS] [--step_inc STEP_INC]

    optional arguments:
      -h, --help           show this help message and exit
      -TFR TFR             single target fill ratio to use in plotting collective decision data
      -sp SP [SP ...]      (optional) sensor probability to use in plotting collective decision data
      <SPECIFIC-HELP-SEE-BELOW>
      --bins BINS          (optional) the number of bins to separate the swarm's decision (default: 2)
      --step_inc STEP_INC  (optional) the increment in simulation steps to evaluate decisions (default: 1000)
    ```
    with specific help for the static version:
    ```
      -U U [U ...]         communications period, communication probability, and number of agents to use in plotting collective decision data
    ```
    and the dynamic version:
    ```
      -U U [U ...]         robot speed and swarm density to use in plotting collective decision data
    ```

    <details><summary>Example: visualize decision plot</summary>

    Suppose we are interested in seeing the collective decisions of robots in the dynamic simulation.

    Specifically, we are interested in seeing the experiment with the following parameters:
    - homogeneous robots with sensor probabilities `{0.525, 0.675, 0.825, 0.975}`, and heterogeneous robots with uniformly distributed sensor probabilities `U(0.525, 0.975)`,
    - environment fill ratio of `0.55`,
    - robot speed of `10` cm/s, and
    - swarm density of `1`,

    with robots choosing from 10 bins at 1000 step increments.

    Then the command is as follows (the heterogeneous sensor probability is encoded as the `identifier`, `min`, `max` in a negative integer; [see the paramater file for more details](parameter_file_setup.md)):
    ```
    $ visualize_multi_agent_data_dynamic.py /home/user/converted_from_pbs.vdg 0.01 -gsi decision -sp -205250975 0.525 0.675 0.825 0.975 -tfr 0.5 -U 10 1
    ```

    <img src="./img/decision_dyn_s10000_t30_tfr550_spd10_den1_bins10.png" width="1000" />

    </details>

### Process Data Manually
First, ensure that the data you need has been converted to `VisualizationDataGroup` binaries.

Once you have all the `*.vdg` files, you can write a python script (maybe a Jupyter notebook) to directly load the `*.vdg` files.

#### Common assumed accuracy `.vdg` files
Let's say you have `A.vdg` that resides in `/home/user/work/`.

The way you load it is
```python
import collective_perception_py.viz_modules as vm

vdg = vm.VisualizarionDataGroupDynamic.load("/home/user/work/A.vdg")
```

Now `vdg` is just the `VisualizationDataGroup` object which contains a member `viz_data_obj_dict`. In this dictionary it has 2 keys, `speed` and `density`, in that order. 

For example, if you want to get the data from the experiment with swarm density of 2 and robot speed of 5.3, you would do `vdg.viz_data_obj_dict[5.3][2]`. This will return a `VisualizationData` object containing all the data given the `sp_range` and `tfr_range` and trials.
```python
vd = vdg.viz_data_obj_dict[5.3][2]
"""Available methods for vd:
vd.AggregateStats(                            vd.detect_convergence(                        vd.num_steps
vd.agg_stats_dict                             vd.get_decision_fractions(                    vd.num_trials
vd.aggregate_statistics(                      vd.get_individual_informed_estimate_metrics(  vd.save(
vd.assumed_sp                                 vd.get_informed_estimate_metrics(             vd.sim_type
vd.comms_period                               vd.load(                                      vd.sp_range
vd.comms_range                                vd.load_pkl_file(                             vd.speed
vd.compute_accuracy(                          vd.load_proto_file(                           vd.stats_obj_dict
vd.density                                    vd.num_agents                                 vd.tfr_range
"""
```

#### Computing convergence time
The `VisualizationData` class has the method `.get_individual_informed_estimate_metrics` which has takes

1. target fill ratio,
2. actual sensor accuracy, and
3. convergence threshold

as arguments. This computes the convergence times and accuracies for each trial; _e.g.,_ if you have 3 trials you will have 3 convergence times and 3 accuracies. That's what the method returns as a tuple.

#### Summary
1. Convert data using the `convert_sim_stats_set_to_viz_data_group.py` script (no need to use a Python session).
2. Load a Python session (script, notebook, etc.) to load the data.
3. (If needed) compute convergence time and then plot however you want.

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