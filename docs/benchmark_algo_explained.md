# Benchmarking algorithms
This page describes how one would execute and implement benchmark algorithms to evaluate collective perception performance. Only dynamic experiments are implemented.

## Description of algorithms
The complete inner workings of the respective algorithms can be found in their cited work. Here, only information related to the implementation in this repository is discussed.

<details><summary><a href="https://ieeexplore.ieee.org/document/8206297">Crosscombe <i>et al.</i> (2017) </a></summary>

All robots start with a random belief state. That is, flawed and non-flawed robots have completely random belief state vectors, where `0` indicate `negative` (falsy), `1` indicate `indeterminate`, and `2` indicate `positive` (truthy). For example, in the case of 5 options, robot 1 can have <0, 1, 1, 0, 0> while robot 2 can have <0, 2, 2, 0, 1>. Note that robot 2's belief will be normalized before being populated and communicated, i.e., robot 2's normalized belief is <0, 1, 1, 0, 0>.

During the updating phase, if the non-flawed robots cannot decide option -- they have multiple indeterminate beliefs, e.g., <1, 1, 0, 1, 0> -- they will randomly pick an option from one of the indeterminate beliefs, while broadcasting the belief. This makes sense because just because a robot randomly picks from options it is uncertain about, it doesn't mean that the robot now is certain about the choice.

</details>

<details><summary><a href="https://ieeexplore.ieee.org/document/9196584">Ebert <i>et al.</i> (2020) </a></summary>

The robots move around in a random walk fashion communicating their observations (or decisions, if positive feedback enabled) to each other. These communicated values are used to adjust the parameters of a Beta distribution, which is the posterior distribution of the tile fill ratio. The robots' decisions states are: `-1` (undecided), `0` (white), and `1` (black). *Note that the black and white decisions are flipped from the original paper to maintain uniformity with our collective perception work.* 

Once each robot has come to their decision, the simulation will be terminated. Otherwise, it will run up until a specified duration in the `.argos` file.

</details>

## Execution
The concepts used in executing simulated experiments is the same as described in the [README](../README.md): a single simulation execution contains multiple experiments, each with multiple trials.

There are, however, differences in running the benchmark experiments.

1. The inner parameter group may be different (although it will always be a pair of parameters). Instead of *target fill ratios* and *sensor probabilities*, the benchmark algorithms may have different pairings.
2. The outer parameter group may be different. Instead of just *robot speed* and *swarm density*, the benchmark algorithms may have additional parameters.

See the dropdowns for specific information related to the benchmark algorithm of interest.

The following subsections describe how to set up the `.argos` configuration files.

### Buzz controller
The location of the `body*` bytecode files depends on where you execute the simulation. For the locally built simulator, it is recommended that you use absolute paths so that the execution location is flexible. For the containerized simulator use the path `/collective_perception/collective_perception_dynamic/build/buzz/body*`.

<details><summary><a href="https://ieeexplore.ieee.org/document/8206297">Crosscombe <i>et al.</i> (2017) </a></summary>

```xml
<buzz_controller_kheperaiv id="bck">

    <!-- Locations of Buzz bytecode files -->
    <params
        bytecode_file="/collective_perception/collective_perception_dynamic/build/buzz/body_crosscombe_2017.bo"
        debug_file="/collective_perception/collective_perception_dynamic/build/buzz/body_crosscombe_2017.bdb" />

</buzz_controller_kheperaiv>
```

</details>

<details><summary><a href="https://ieeexplore.ieee.org/document/9196584">Ebert <i>et al.</i> (2020) </a></summary>

```xml
<buzz_controller_kheperaiv id="bck">

    <!-- Locations of Buzz bytecode files -->
    <params
        bytecode_file="/collective_perception/collective_perception_dynamic/build/buzz/body_ebert_2020.bo"
        debug_file="/collective_perception/collective_perception_dynamic/build/buzz/body_ebert_2020.bdb" />

</buzz_controller_kheperaiv>
```

</details>

### Loop functions
For the location of the `benchmarking_loop_functions` library, specify them as you would the Buzz bytecode files: an absolute path for the local build or `/collective_perception/collective_perception_dynamic/build/...` for the container. The rest of the parameters are explained in the snippet below. The general `<benchmarking />` parameters are filled as the following, with specific benchmark algorithm parameters described further below.

```xml
<loop_functions library="/collective_perception/collective_perception_dynamic/build/src/libbenchmarking_loop_functions" label="benchmarking_loop_functions">

    <benchmarking>
        <!-- Specific benchmarking algorithm parameters -->
        <algorithm ... />

        <!-- Number of tiles for the arena in the x and y direction -->
        <!-- NOTE: must have equal number of tile counts -->
        <arena_tiles tile_count_x="1000" tile_count_y="1000" />

        <!-- Range of target fill ratios between `min` and `max` of `steps` increments -->
        <!-- NOTE: must be between 0.0 to 1.0 -->
        <fill_ratio_range min="0.05" max="0.95" steps="19" />

        <!-- Robot speed in cm/s -->
        <speed value="10.0" />

        <!-- Number of trials for a specific fill ratio and sensor probability -->
        <num_trials value="5" />

        <!-- Path to the output data and datetime in filename -->
        <!-- Note: the extensions ".json" must be retained -->
        <path folder="data"
              data="data.json"
              include_datetime="true" />

        <!-- Verbosity level -->
        <!-- Options: "full", "reduced", "none" -->
        <verbosity level="reduced" />

    </benchmarking>

</loop_functions>
```

<details><summary><a href="https://ieeexplore.ieee.org/document/8206297">Crosscombe <i>et al.</i> (2017) </a></summary>

```xml
<algorithm type="crosscombe_2017"> <!-- the value to `type` is provided as a macro in the benchmark algorithm `.hpp` file -->

    <!-- Number of options for robots to choose from -->
    <num_possible_options value="10" />

    <!-- Range of flawed robot ratios to simulate in each experiment -->
    <flawed_robot_ratio_range min="0.1" max="0.5" steps="2" />
</algorithm>
```

For this benchmark algorithm, the outer parameter group is unchanged (robot speed and swarm density). The inner group parameters are `fill_ratio_range` and `sensor_probability_range`.

*Note: while arena floor has tile colors, this benchmark algorithm doesn't actually detect tile colors. What the arena floor look like does not actually affect the robot performance; it is however a reflection of the fill ratio.*

</details>

<details><summary><a href="https://ieeexplore.ieee.org/document/9196584">Ebert <i>et al.</i> (2020) </a></summary>

```xml
<algorithm type="ebert_2020"> <!-- the value to `type` is provided as a macro in the benchmark algorithm `.hpp` file -->

    <!--
        range of sensor probabilities
        `steps` must be an integer:
            - positive, probabilities are spread linearly from min to max
            - -2 indicates a uniform distribution with range [`min`, `max`)
            - -3 indicates a normal distribution with mean=`min`, variance=`max`
    -->
    <sensor_probability_range min="0.525" max="0.975" steps="-2" />

    <!-- Flag to activate positive feedback -->
    <positive_feedback bool="true" />

    <!-- Beta prior distribution parameters (float value a that initializes both parameters ==> Beta(a, a) -->
    <prior value="1" />

    <!-- Credible threshold for robots to make a decision -->
    <credible_threshold value="0.9" />
</algorithm>
```

For this benchmark algorithm, the additional parameters in the outer parameter group are `positive_feedback`, `prior`, and `credible_threshold`. `sensor_probability_range` remains a parameter in the inner parameter group.

</details>

## Analysis
A single data file will be generated for each completed trial. All data files for benchmark algorithms would be in `.json` format. The data files vary depending on the benchmark algorithm type, with the following as the only common output.

```json
{
    "sim_type": "<BENCHMARK-AUTHOR>_<BENCHMARK-YEAR>",  // benchmark algorithm identifier (string); provided as a macro in the benchmark algorithm `.hpp` file
    "num_agents": 25,                                   // total number of robots (int)
    "num_trials": 2,                                    // total number of trials (int)
    "num_steps": 100,                                   // total number of time steps (int)
    "comms_range": 0.699999988079,                      // communication range of robot in meters (float)
    "speed": 14.140000343322754,                        // straight line speed of robot in m/s (float)
    "density": 10.00001049041748,                       // density of robot swarm (float); formula is shown in scripts_explained.md
    "tfr": 0.95,                                        // target fill ratio of the environment
    "trial_ind": 1,                                     // trial index for this data file
    /*
    ...                                                 // benchmark algorithm specific data
    */
}
```

Then, to visualize the data files, we use the `visualize_multi_agent_data_benchmark.py` script:
```
usage: visualize_multi_agent_data_benchmark.py [-h] [--step_inc STEP_INC] [-s] FOLDER TFR {crosscombe_2017} ...

Visualize benchmark data

positional arguments:
  FOLDER               path to folder containing the JSON data files
  TFR                  single target fill ratio to use in plotting collective decision data
  {crosscombe_2017}    benchmark type
    crosscombe_2017    crosscombe_2017 benchmark

optional arguments:
  -h, --help           show this help message and exit
  --step_inc STEP_INC  (optional) the increment in simulation steps to evaluate decisions (default: 1000)
  -s                   flag to show the plots
```
Benchmark algorithm-specific arguments are described in their respective dropdowns.

<details><summary><a href="https://ieeexplore.ieee.org/document/8206297">Crosscombe <i>et al.</i> (2017) </a></summary>

JSON data:
```json
{
    "sim_type": "crosscombe_2017",  // benchmark algorithm identifier (string)
    /* 
    ...                             // common data output
    */
    "frr": 0.1,                     // flawed robot ratio in this trial (float)
    "num_flawed_robots": 3,         // number of flawed robots in this trial (int); is the rounded value of frr*num_agents
    "option_qualities": [           // quality of the options (array of int); dictates broadcast duration in units of time steps
        0,
        0,
        0,
        2,
        7,
        23,
        54,
        101,
        147,
        166
    ],
    "beliefs": [                    // belief states (array of array of string)
        [                           // belief state of robot 0 (array of string)
            "0001000010",           // belief state of robot 0 at time step = 0 (string)
            "0001000010",           // belief state of robot 0 at time step = 1 (string)
            "0001000100",
            "0001000100",
            "0001000100"
        ],
        [                           // belief state of robot 1
            "0001010100",           // belief state of robot 1 at time step = 0 (string)
            "0001010100",
            "0001010100",
            "0001000100",
            "0001000100"
        ],
        [
            "0101000101",
            "0100000100",
            "0100000100",
            "0100000100",
            "0100000100"
        ],
        [
            "1010000100",
            "1010000100",
            "1010000100",           // belief state of robot 3 at time step = 2 (string)
            "1010000100",
            "1010000100"
        ]
    ]
}
```

Visualization script:
```
usage: visualize_multi_agent_data_benchmark.py FOLDER TFR crosscombe_2017 [-h] {decision,series} ...

positional arguments:
  {decision,series}  commands for visualization type
    decision         visualize collective-decision making data
    series           visualize time series data

optional arguments:
  -h, --help         show this help message and exit
```

- `decision` usage:
    ```
    usage: visualize_multi_agent_data_benchmark.py FOLDER TFR crosscombe_2017 decision [-h] FRR [FRR ...]

    positional arguments:
    FRR         flawed robot ratio (space-delimited array) to use in plotting collective decision data

    optional arguments:
    -h, --help  show this help message and exit
    ```

- `series` usage:
    ```
    usage: visualize_multi_agent_data_benchmark.py FOLDER TFR crosscombe_2017 series [-h] FRR

    positional arguments:
    FRR         flawed robot ratio to use in plotting time series data

    optional arguments:
    -h, --help  show this help message and exit
    ```
</details>

<details><summary><a href="https://ieeexplore.ieee.org/document/9196584">Ebert <i>et al.</i> (2017) </a></summary>

JSON data:
```json
{
    "sim_type": "ebert_2020",       // benchmark algorithm identifier (string)
    /* 
    ...                             // common data output
    */
    "sp": 0.675,                    // sensor probability in this trial (float)
    "prior_param": 10,              // prior distribution parameters, i.e., BetaDist(alpha=prior_param, beta=prior_param) (int)
    "credible_threshold": 0.99,     // credible threshold (float)
    "positive_feedback": false,     // positive feedback flag (bool)
    "collectively_decided": false,  // flag that indicates whether the robots have each made a decision (bool)
    "data_str": [                   // data string array of arrays; data string has the form "<A>,<B>,<P>,<D>" where
                                    //      <A> = alpha parameter of Beta distribution
                                    //      <B> = beta parameter of Beta distribution
                                    //      <P> = P(X < 0.5) where X ~ BetaDist(alpha=<A>, beta=<B>)
                                    //      <D> = decision made by the robot
        [                           // data string of robot 0 (array of string)
            "10,11,0.588099,-1",    // data string of robot 0 at time = 0 (string)
            "13,12,0.419410,-1",    // data string of robot 0 at time = 1 (string)
            "15,14,0.425277,-1",
            "16,17,0.569975,-1",
            "18,19,0.566030,-1",
        ],
        [                           // data string of robot 1
            "11,10,0.411901,-1",    // data string of robot 1 at time step = 0 (string)
            "12,12,0.500000,-1",
            "15,12,0.278599,-1",
            "15,15,0.500000,-1",
            "16,17,0.569975,-1",
        ],
        [
            "11,10,0.411901,-1",
            "12,13,0.580590,-1",
            "12,17,0.827536,-1",
            "15,18,0.701693,-1",
            "16,21,0.797484,-1",
        ],
        [
            "10,11,0.588099,-1",
            "11,13,0.661180,-1",
            "11,16,0.836530,-1",    // data string of robot 3 at time step = 2 (string)
            "12,18,0.867535,-1",
            "14,19,0.811457,-1",
        ]
    ]
}
```

Visualization script:
```
lorem ipsum
```
</details>

## Development
This subsection describes the main action items -- *should be exhaustive, but some items may not have been included* -- needed to implement benchmark algorithms correctly.

1. Benchmark algorithms can be implemented by creating the following files.
    - `include/collective_perception_cpp/benchmark_<BENCHMARK-AUTHOR>_<BENCHMARK-YEAR>.hpp` that contains the following includes and macros (modify the macro values as needed):
        ```cpp
        #include "benchmark_algorithm.hpp"

        // Define default benchmark algorithm identifiers; if modified then must change at other locations
        #define <BENCHMARK-AUTHOR>_<BENCHMARK-YEAR> std::string("<BENCHMARK-AUTHOR>_<BENCHMARK-YEAR>")
        #define <BENCHMARK-AUTHOR>_<BENCHMARK-YEAR>_PARAM std::string("<DESIRED-PARAMETER-TO-TEST>")
        #define <BENCHMARK-AUTHOR>_<BENCHMARK-YEAR>_PARAM_ABBR std::string("<DESIRED-PARAMETER-ABBREVIATED>")
        ```
    - `src/benchmark_<BENCHMARK-AUTHOR>_<BENCHMARK-YEAR>.cpp`

2. In the `.hpp` file, create a structure `BenchmarkData<BENCHMARK-AUTHOR><BENCHMARK-YEAR>` to store benchmark algorithm data that is derived from `BenchmarkDataBase` and a class `Benchmark<BENCHMARK-AUTHOR><BENCHMARK-YEAR>` that is derived from `BenchmarkAlgorithmTemplate<BenchmarkData<BENCHMARK-AUTHOR><BENCHMARK-YEAR>>`. For example, if the author and year are `Dummy` and `2016`, then the structure and class would look like:
    ```cpp
    struct BenchmarkDataDummy2016 : BenchmarkDataBase
    {
        ...
    };

    class BenchmarkDummy2016 : public BenchmarkAlgorithmTemplate<BenchmarkDataDummy2016>
    {
        ...
    };
    ```
    See `benchmark_crosscombe_2017.hpp` as an example.

3. In `include/collective_perception_cpp/benchmarking_loop_functions.hpp`, include the created benchmark algorithm header file: `#include "benchmark_<BENCHMARK-AUTHOR>_<BENCHMARK-YEAR>.hpp"`.

4. Append the source file to the `add_library` directive for the `benchmarking_loop_functions` library in `src/CMakeLists.txt`:
    ```cmake
    add_library(benchmarking_loop_functions
        SHARED
        benchmarking_loop_functions.cpp
        ...
        benchmark_<BENCHMARK-AUTHOR>_<BENCHMARK-YEAR>.cpp
    )
    ```

5. In `benchmarking_loop_functions.*pp`, update `BenchmarkingLoopFunctions::InitializeBenchmarkAlgorithm(TConfigurationNode)` to correctly initialize the benchmark algorithm. In the simplest case, all you have to do is append an `else if` statement. For example, if the author and year are `Dummy` and `2016`, then the updated `if` statement looks like the following.
    ```cpp
    if (algorithm_str_id_ == CROSSCOMBE_2017)
    {
        benchmark_algo_ptr_ =
            std::make_shared<BenchmarkCrosscombe2017>(buzz_foreach_vm_func, t_tree, robot_id_vec);
    }
    else if (algorithm_str_id_ == DUMMY_2016)
    {
        benchmark_algo_ptr_ =
            std::make_shared<BenchmarkDummy2016>(buzz_foreach_vm_func, t_tree, robot_id_vec);
    }
    else
    {
        THROW_ARGOSEXCEPTION("Unknown benchmark algorithm!");
    }

    benchmark_algo_ptr_->Init();
    ```
6. Ensure that you are writing JSON data files. The `nlohmann::json` library has been incorporated in this repository, see `benchmark_crosscombe_2017.*pp` as examples.
7. Create the Buzz controller bytecode file `body_<BENCHMARK-AUTHOR>_<BENCHMARK-YEAR>.bzz` (*e.g.,* `body_dummy_2016.bzz`). You can include `body_common.bzz` to use the common body functions. See `body_crosscombe_2017.bzz` as an example. Then add the line `buzz_make(body_<BENCHMARK-AUTHOR>_<BENCHMARK-YEAR>.BZZ INCLUDES body_common.bzz)` to `buzz/CMakeLists.txt`.
8. Implement the visualization functions in `viz_modules_benchmark.py` by creating a class that is derived from `BenchmarkVisualizerBase(ABC)` (for example, `Dummy2016Visualizer(BenchmarkVisualizerBase)`). See `viz_modules_benchmark.py` for more details and examples.