# How to setup parameter files
## Static topology simulation
Using the template from `examples/param/param_multi_agent_sim_static.yaml`, fill in the desired parameter values.
```yaml
--- # document start marker
numAgents: <INT>      # number of agents
commsGraph:
  type: <STRING>      # communications graph type: full, line, ring, scale-free
  commsPeriod: <INT>  # number of steps per communication instance
  commsProb: <FLOAT>  # communication probability (only 1.0 is supported currently)
sensorProb:           # sensor probability/accuracy range (inclusive) to simulate
  min: <FLOAT>        # min sensor probability
  max: <FLOAT>        # max sensor probability
  incSteps: <INT>     # number of steps from min to max inclusive
                      # set to -2 for uniform distribution with range [`min`, `max`)
                      # set to -3 for normal distribution with (mean=`min`, variance=`max`)
targFillRatios:       # target fill ratio range (inclusive) to simulate
  min: <FLOAT>        # min desired fill ratio
  max: <FLOAT>        # max desired fill ratio
  incSteps: <INT>     # number of steps from min to max inclusive
numTrials: <INT>      # number of trials to repeat for each `targFillRatio` & `sensorProb` set
numSteps: <INT>       # number of steps (each observation is made per step)
legacy: <BOOL>        # flag to use legacy equations
... # document end marker
```
Save the file with the name `param_multi_agent_sim_static.yaml` at the directory where you will be executing the simulation.

## Dynamic topology simulation
The parameter file used for the dynamic simulation is the same as the configuration file used by ARGoS, which is described in their [official documentation](https://www.argos-sim.info/user_manual.php). Here, only parameters specifically related to the simulated experiments will be discussed. You can find the template from `examples/param/param_multi_agent_sim_dynamic.argos`. *For information on benchmark algorithms (how to run, implement, and analyze), see [here](./benchmark_algo_explained.md).*

### Number of steps
An ARGoS "experiment" that is repeated `m` times means that it has `m` trials. Thus `length * ticks_per_second` of an `experiment` becomes the number of steps for a single trial.
```xml
<!-- Duration of a single trial -->
<experiment length="500" ticks_per_second="10" random_seed="0" />
```

### Buzz controller
The location of the `body.*` bytecode files depends on where you execute the simulation. For the locally built simulator, it is recommended that you use absolute paths so that the execution location is flexible. For the containerized simulator use the path `/collective_perception/collective_perception_dynamic/build/buzz/body.*`.
```xml
<buzz_controller_kheperaiv id="bck">

    <!-- Locations of Buzz bytecode files -->
    <params
        bytecode_file="/collective_perception/collective_perception_dynamic/build/buzz/body.bo"
        debug_file="/collective_perception/collective_perception_dynamic/build/buzz/body.bdb" />

</buzz_controller_kheperaiv>
```

### Loop functions
For the location of the `collective_perception_loop_functions` library, specify them as you would the Buzz bytecode files: an absolute path for the local build, `/collective_perception/collective_perception_dynamic/build/...` for the container. The rest of the parameters are explained in the snippet below. For further information on the DAC plugin, see [here](dac_plugin_explained.md)
```xml
<loop_functions library="/collective_perception/collective_perception_dynamic/build/src/libcollective_perception_loop_functions" label="collective_perception_loop_functions">

    <collective_perception>

        <!-- Number of tiles for the arena in the x and y direction -->
        <!-- NOTE: must have equal number of tile counts -->
        <arena_tiles tile_count_x="1000" tile_count_y="1000" />

        <!-- Range of target fill ratios between `min` and `max` of `steps` increments -->
        <!-- NOTE: must be between 0.0 to 1.0 -->
        <fill_ratio_range min="0.05" max="0.95" steps="19" />

        <!-- Range of sensor probabilities between `min` and `max` of `steps` increments if `steps is a positive integer; otherwise:
            - `steps` = -2 indicates a uniform distribution with range [`min`, `max`)
            - `steps` = -3 indicates a normal distribution with mean=`min`, variance=`max`
        -->
        <sensor_probability_range min="0.525" max="0.975" steps="19" />

        <!-- Robot speed in cm/s -->
        <speed value="10.0" />

        <!-- Number of trials for a specific fill ratio and sensor probability -->
        <num_trials value="5" />

        <!-- Disable n random robots at specified time in seconds; disabled robots would stop completely and retain its last values -->
        <!-- Note: `amount` = 0 to prevent any disabling of robots -->
        <robot_disabling amount="5" sim_clock_time="50" >
            <motion_disable bool="true" />
            <comms_disable bool="true" />
            <sense_disable bool="true" />
        </robot_disabling>

        <!-- Path to the output data and datetime in filename -->
        <!-- Note: the extensions ".pbs" and ".pbad" must be retained -->
        <path folder="data"
              stats="multi_agent_sim_dynamic_stats.pbs"
              agent_data="multi_agent_sim_dynamic_agent_data.pbad"
              include_datetime="true" />

        <!-- Verbosity level -->
        <!-- Options: "full", "reduced", "none" -->
        <verbosity level="reduced" />

        <!-- Legacy equations -->
        <!-- Options: "true", "false" -->
        <legacy bool="false" />

        <!-- DAC plugin usage: writes to a CSV file that acts as an interface -->
        <!-- Options: "true", "false" -->
        <dac_plugin activate="false">
            <!--
                num_bins: number of bins to partition the fill ratio; e.g., num_bins=2 means the robots choose between 2 bins, f<0.5 or f >0.5.
                write_period: csv file writing period in seconds; use -1 if only write once at the end of each trial
                csv_path: path to csv file
            -->
            <param num_bins="2" write_period="10" csv_path="/home/khaiyichin/test.csv" />
        </dac_plugin>

    </collective_perception>

</loop_functions>
```
**Note: it is advised that the extensions `.pbs` and `.pbad` be used for the `stats` and `agent_data` files respectively.**

### Arena size and swarm density
The arena size can be modified as explained in the official ARGoS documentation. The effect it has on the generated tile size is described by the equation `arena_size_*/arena_tile_count_*` for both the x and y directions.

The swarm density is modified indirectly through the positions of the 4 walls that constrain the robots' movable space. Use the `compute_wall_positions` script (see [here](scripts_explained.md)) to obtain the appropriate values based on the desired swarm density.
```xml
<arena size="10, 10, 1" center="0,0,0.5">

    <!-- Method to generate the arena floor; NO MODIFICATION NEEDED -->
    <floor id="floor" source="loop_functions" pixels_per_meter="100" />

    <!-- Location of four walls to constrain the robots' movable space -->
    <box id="wall_north" size="10,0.1,0.5" movable="false">
        <body position="0,4.4365995,0" orientation="0,0,0" />
    </box>
    <box id="wall_south" size="10,0.1,0.5" movable="false">
        <body position="0,-4.4365995,0" orientation="0,0,0" />
    </box>
    <box id="wall_east" size="0.1,10,0.5" movable="false">
        <body position="4.4365995,0,0" orientation="0,0,0" />
    </box>
    <box id="wall_west" size="0.1,10,0.5" movable="false">
        <body position="-4.4365995,0,0" orientation="0,0,0" />
    </box>

    <distribute>

        <!-- Robot placement distribution -->
        <position method="uniform"
                  min="-4.4365995,-4.4365995,0"
                  max="4.4365995,4.4365995,0" />
        <orientation method="uniform" min="0,0,0" max="360,0,0" />

        <!-- Number of robots (modify the `quantity` attribute); NO OTHER MODIFICATIONS NEEDED -->
        <entity quantity="50" max_trials="100" base_num="0">
            <kheperaiv id="kiv" rab_data_size="50" rab_range="0.7">
                <controller config="bck" />
            </kheperaiv>
        </entity>

    </distribute>

</arena>
```

## Additional notes
The `legacy` parameter allows you to switch between using legacy or updated social equations.

With the $i$-th agent having the set of $\mathcal N_i$ neighbors at timestep $k$, the *legacy* social estimate $\bar{x}$ and confidence $\beta$ are defined as

$$ \bar{x}^k_i = \frac{1}{||\mathcal N_i||} \sum_{j \in \mathcal N_i} \hat{x}^k_j,$$

$$ \beta^k_i = \bigg[ \frac{1}{||\mathcal N_i||} \sum_{j \in \mathcal N_i} \frac{1}{\alpha^k_j} \bigg]^{-1} $$

where $\hat{x}$ is the local estimate and $\alpha$ is the local confidence.

On the other hand, the *updated* social estimate and confidence are defined as

$$ \bar{x}^k_i = \frac{1}{\sum_{j \in \mathcal N_i} \alpha^k_j} \sum_{j \in \mathcal N_i} \alpha^k_j \hat{x}^k_j, $$

$$ \beta^k_i = \sum_{j \in \mathcal N_i} \alpha^k_j. $$
