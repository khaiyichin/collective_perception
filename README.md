# Collective Perception with Imperfect Sensors
## Introduction
This repository provides the code to simulate collective perception experiments shown in [Collective Perception with Imperfect Sensors (work-in-progress)]().

Two simulators are provided here:
1. Python-based static topology simulator `collective_perception_static`, and
2. ARGoS-based dynamic topology simulator `collection_perception_dynamic`.

### Static topology simulation
The robots in the static topology simulator do not move in the normal sense and have fixed communication channels with its neighbors (the communication network is specified by the user). In each simulated experiment, a fixed number of robots would traverse their own black and white tile-track and communicate periodically with their neighbors. The figure below illustrates what it would look like for 4 robots in a ring topology.

<img src="static_sim_graphic.png" alt="Static simulated experiment visualized" width="450"/>

### Dynamic topology simulation
The robots in the dynamic topology simulator move around a square arena of black and white tiles. In each simulated experiment, a fixed number of robots would randomly move around the arena and communicate with neighbors within their proximity. The figure below is a screenshot of the simulation for 25 robots.

<img src="dynamic_sim_graphic.png" alt="Dynamic simulated experiment visualized" width="450"/>

## Requirements
### Local build
- Python 3.8+ and `pip`
- CMake 3.15+
- [ARGoS](https://github.com/ilpincy/argos3.git) - *version used: [da33b87](https://github.com/ilpincy/argos3/tree/da33b8786293dad40307e4182bb791add1e89172)*
- [Buzz](https://github.com/NESTLab/Buzz) - *version used: [6a9a51f](https://github.com/NESTLab/Buzz/tree/6a9a51f9b658b76fc995152546d4b625e74abb6d)*
- [Protobuf v21.1+ (`proto3`)](https://github.com/protocolbuffers/protobuf.git) - *source build recommended, although the `apt` package version may work as well*
- [GraphTool v2.45+](https://graph-tool.skewed.de/)

### Container
- [Apptainer v1.0.2+](https://github.com/apptainer/apptainer)

## Installation
### Local build
The following instructions were tested on Ubuntu 20.04 LTS Focal Fossa. It's likely they would work on MacOS and Windows with some modification.
1. Ensure that all requirements are satisfied.
2. Clone the repository and go to the root directory.
    ```
    $ git clone https://github.com/khaiyichin/collective_perception.git
    $ cd collective_perception/
    ```
3. Run the `install` script (it builds `collective_perception_dynamic` and `collective_perception_static` under the hood).
    ```
    $ ./install.sh
    ```

### Container
The reason for using `apptainer` is the benefit of rootless execution (which is a problem for Docker containers), which permits usage of the simulators on HPC clusters. Another reason is due to the usage of GraphTool for the static simulator &mdash; installed using the `apt` package manager &mdash; which requires `sudo` privileges.

To create simulator containers, only `apptainer` is required; the other requirements will be installed as the containers are built. The containers are built in stages so that you can modify the `collective_perception*` source code anytime and only build the last level without having to start from the top.

1. Create a `containers` directory in the `apptainer` directory.
    ```
    $ cd apptainer && mkdir containers
    ```
2. Build the first container, which provides the ARGoS and Buzz base.
    ```
    $ sudo apptainer build argos_buzz_no_qt_base.sif ../def/argos_buzz_no_qt_base.def
    ```
    When the build finishes you should see the `argos_buzz_no_qt_base.sif` container.
3. Build the second container on top of the previous container (specified in the definition file), which provides the Protobuf layer.
    ```
    $ sudo apptainer build protobuf_no_qt_layer.sif ../def/protobuf_no_qt_layer.def
    ```
    When the build finishes you should see the `protobuf_no_qt_layer.sif` container.
4. Build the final layer.
    ```
    $ sudo apptainer build multi_agent_sim_full_no_qt.sif ../def/multi_agent_sim_full_no_qt.sif
    ```
    When the build finishes you should see the multi_agent_sim_full_no_qt.sif container.

## Execution
The instructions here describe scripts that provide a **single simulation execution**. In a single simulation execution, there can be multiple experiments, each of which may have repeated trials.
```mermaid
graph TD;
    s[Single simulation execution]-->e1[Experiment 1];
    s-->e2[Experiment 2];
    s-->en[Experiment n];
    e1-->t11[Trial 1];
    e1-->t12[Trial 2];
    e1-->t1m[Trial m];
    e2-->t21[Trial 1];
    e2-->t22[Trial 2];
    e2-->t2m[Trial m];
    en-->tn1[Trial 1];
    en-->tn2[Trial 2];
    en-->tnm[Trial m];
```
In general, simulation executions are controlled by two groups of parameters: an outer group and an inner group. The *outer parameter group is fixed for a single simulation execution*, i.e., the parameter values stay the same for all experiment and trials. The *inner parameter group is fixed only within an experiment*, i.e., the parameter values stay the same between the repeated trials of a single experiment, but vary across experiments.

The inner parameter group has two parameters: *target fill ratios* and *sensor probabilities*, and is the same for both simulation types. That is, a pair of target fill ratio and sensor probability values are used in one experiment. When the experiment (including the repeated trials) completes, a different pair of target fill ratio and sensor probability values is used in the next experiment.

The outer parameter group differs between the static and dynamic simulation types. For the static topology simulator, the parameters are *communication period* and *number of agents*; for the dynamic topology simulator, the parameters are *robot speed* and *swarm density*.

The following instructions apply directly for the local build; for the container simulator simply prepend `apptainer exec multi_agent_sim_full_no_qt.sif` to the commands ([see the Apptainer documentation for more info](https://apptainer.org/docs/)).

### Static topology simulation
1. Set up the desired experimental parameters according as shown [here](docs/parameter_file_setup.md).
2. Activate the Python virtual environment (created in the `collective_perception_static` directory).
    ```
    $ cd collective_perception_static/
    $ source .venv/bin/activate
    ```
    *When running the container simulator, you can skip this step; there's no need to activate any virtual environment since the Python modules are installed directly to the container.*
3. Run the simulation (ensure that the parameter file is present in the execution directory).
    ```
    $ multi_agent_sim_static.py
    ```
    Script help:
    ```
    usage: multi_agent_sim_static.py [-h] [-p]

    Execute multi-agent simulation with static topologies.

    optional arguments:
      -h, --help  show this help message and exit
      -p          flag to use cores to run simulations in parallel
    ```
4. When the execution completes, it will output pickled data in newly created directory `data`. (Dev note: pickles may be updated to protobufs; output directory structure may be modified.)

### Dynamic topology simulation
1. Set up the desired experimental parameters according as shown [here](docs/parameter_file_setup.md).
2. Run the simulation (ensure that the parameter file is present in the execution directory).
    ```
    $ run_dynamic_simulations -c param_multi_agent_sim_dynamic.argos
    ```
3. When the execution completes, it will output protobuf files in a local directory specified in your configuration file.

## Visualization of experiment data
Once the simulation execution completes, you will need to convert the output data into a `VisualizationDataGroup` type data file. The workflow looks like the following:
```mermaid
    graph LR;
    a(Execute simulated experiments) --> b(Obtain experiment data) --> c(Convert experiment data to visualization data) --> d(Visualize data);
```
See the [here](docs/scripts_explained.md) for detailed instructions, but the general idea is:
1. Convert the experiment data with the `convert_exp_data_to_viz_data_group.py` script (static) or `convert_sim_stats_set_to_viz_data_group.py` script (dynamic).
2. Visualize the converted data with the `visualize_multi_agent_data_*.py`.

## Testing
Unit tests have been provided to aid any updates to the source code. Besides identifying the kinds of testing imposed on the source code, looking into the test files can help you understand how the algorithm works.

For the static simulator, simply do the following to run tests after the virtual environment has been setup.
```
$ cd collective_perception_static
$ source .venv/bin/activate
$ pytest --workers=<N> # run the test in parallel using N cores; remove the "--workers" flag if sequential testing is desired
```

For the dynamic simulator, you will need to clone this repository including its submodules, notably the [Catch2](https://github.com/catchorg/Catch2/) repository in `collective_perception_dynamic/extern/`.
```
$ git clone --recurse-submodules https://github.com/khaiyichin/collective_perception.git
```
Build the dynamic simulator project in `Debug` mode.
```
$ cd collective_perception_dynamic
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Debug ..
...
$ make -j$(nproc)
```
Then you can run the tests by either doing `make test` or executing them with the command line.
```
# Run without any flags, see the Catch2 documentation for more info
$ tests/tests

# Run with gdb
$ gdb tests/tests # or gdb --args tests/tests <ARGS> if you have arguments, e.g., Catch2 flags

# Run with valgrind
$ valgrind tests/tests <ARGS-IF-ANY>
```