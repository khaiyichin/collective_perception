# Interacting with the DAC plugin CSV file interface

## Overview
To collect live simulation statistics, the DAC plugin can be activated, though this will likely slow the simulation down slightly. Instructions on how to activate it in the `.argos` parameter file is detailed [here](parameter_file_setup.md).

By activating it in the `.argos` file, a `.csv` file will be created. You will be able to choose the path of the `.csv` file and how often it's being written to.

The file acts as an interface/log for the ARGoS simulation that you can read from. Presently, the only (running) statistics output are
* the number of active robots at log time,
* the number of disabled robots at log time, and
* the fraction of correct decisions made by the active robots at log time.
Note that the fraction of correct decisions is affected by the `num_bins` argument that you set in the `.argos` file.

## Description
An example of the `.csv` file would look like the following if opened in a text editor:
```
121222_075629,experimentstats
,range,0.700000
,speed,10.000000
,density,5.500000,area,3.078761
,robots,11
,fillratio,0.950000
,sensorprob,0.525000
121222_075629,trialindex,0
121222_075631,trialstats
,simseconds,50
,active,11
,disabled,0
,fractioncorrectdecisions,0.636364
121222_075632,trialstats
,simseconds,100
,active,6
,disabled,5
,fractioncorrectdecisions,0.333333
121222_075632,trialindex,1
121222_075633,trialstats
,simseconds,50
,active,11
,disabled,0
,fractioncorrectdecisions,0.363636
121222_075635,trialstats
,simseconds,100
,active,6
,disabled,5
,fractioncorrectdecisions,0.500000
121222_075635,trialindex,2
121222_075636,trialstats
,simseconds,50
,active,11
,disabled,0
,fractioncorrectdecisions,0.363636
121222_075638,trialstats
,simseconds,100
,active,6
,disabled,5
,fractioncorrectdecisions,1.000000
121222_075638,experimentstats
,range,0.700000
,speed,10.000000
,density,5.500000,area,3.078761
,robots,11
,fillratio,0.950000
,sensorprob,0.975000
121222_075638,trialindex,0
121222_075640,trialstats
,simseconds,50
,active,11
,disabled,0
,fractioncorrectdecisions,1.000000
121222_075642,trialstats
,simseconds,100
,active,6
,disabled,5
,fractioncorrectdecisions,1.000000
121222_075642,trialindex,1
121222_075645,trialstats
,simseconds,50
,active,11
,disabled,0
,fractioncorrectdecisions,1.000000
121222_075646,trialstats
,simseconds,100
,active,6
,disabled,5
,fractioncorrectdecisions,1.000000
121222_075646,trialindex,2
121222_075648,trialstats
,simseconds,50
,active,11
,disabled,0
,fractioncorrectdecisions,1.000000
121222_075650,trialstats
,simseconds,100
,active,6
,disabled,5
,fractioncorrectdecisions,1.000000
121222_075650,experimentcomplete
```
The same file would look like the following if you opened it in MS Excel or Google Sheets:
||||||
| --- | --- | --- | --- | --- |
| 121222_075629 | experimentstats |
|  | range | 0.700000 |
|  | speed | 10.000000 |
|  | density | 5.500000 | area | 3.078761 |
|  | robots | 11 |
|  | fillratio | 0.950000 |
|  | sensorprob | 0.525000 |
| 121222_075629 | trialindex | 0 |
| 121222_075631 | trialstats |
|  | simseconds | 50 |
|  | active | 11 |
|  | disabled | 0 |
|  | fractioncorrectdecisions | 0.636364 |
| 121222_075632 | trialstats |
|  | simseconds | 100 |
|  | active | 6 |
|  | disabled | 5 |
|  | fractioncorrectdecisions | 0.333333 |
| 121222_075632 | trialindex | 1 |
| 121222_075633 | trialstats |
|  | simseconds | 50 |
|  | active | 11 |
|  | disabled | 0 |
|  | fractioncorrectdecisions | 0.363636 |
| 121222_075635 | trialstats |
|  | simseconds | 100 |
|  | active | 6 |
|  | disabled | 5 |
|  | fractioncorrectdecisions | 0.500000 |
| 121222_075635 | trialindex | 2 |
| 121222_075636 | trialstats |
|  | simseconds | 50 |
|  | active | 11 |
|  | disabled | 0 |
|  | fractioncorrectdecisions | 0.363636 |
| 121222_075638 | trialstats |
|  | simseconds | 100 |
|  | active | 6 |
|  | disabled | 5 |
|  | fractioncorrectdecisions | 1.000000 |
| 121222_075638 | experimentstats |
|  | range | 0.700000 |
|  | speed | 10.000000 |
|  | density | 5.500000 | area | 3.078761 |
|  | robots | 11 |
|  | fillratio | 0.950000 |
|  | sensorprob | 0.975000 |
| 121222_075638 | trialindex | 0 |
| 121222_075640 | trialstats |
|  | simseconds | 50 |
|  | active | 11 |
|  | disabled | 0 |
|  | fractioncorrectdecisions | 1.000000 |
| 121222_075642 | trialstats |
|  | simseconds | 100 |
|  | active | 6 |
|  | disabled | 5 |
|  | fractioncorrectdecisions | 1.000000 |
| 121222_075642 | trialindex | 1 |
| 121222_075645 | trialstats |
|  | simseconds | 50 |
|  | active | 11 |
|  | disabled | 0 |
|  | fractioncorrectdecisions | 1.000000 |
| 121222_075646 | trialstats |
|  | simseconds | 100 |
|  | active | 6 |
|  | disabled | 5 |
|  | fractioncorrectdecisions | 1.000000 |
| 121222_075646 | trialindex | 2 |
| 121222_075648 | trialstats |
|  | simseconds | 50 |
|  | active | 11 |
|  | disabled | 0 |
|  | fractioncorrectdecisions | 1.000000 |
| 121222_075650 | trialstats |
|  | simseconds | 100 |
|  | active | 6 |
|  | disabled | 5 |
|  | fractioncorrectdecisions | 1.000000 |
| 121222_075650 | experimentcomplete |

* The 1st column is reserved for the timestamp in `monthdayyear_HOURMINUTESECOND` form.
* The 2nd column is the place where you want to keep track of:
    * The row with `experimentstats` marks a new experiment, with the rows immediately following it indicating the experimental parameters.
    * The row with `trialindex` marks a new trial. All trial statistics after the current row belong to the current trial, until another row with `trialindex` is printed.
    * The row with `trialstats` marks the section where the trial statistics are printed in the rows immediately following it.
    * The row with `experimentcomplete` marks the end of the entire simulated experiment.
* The 3rd column is the place where the values for the 2nd column (if applicable) is printed.
* The 4th column is reserved for additional parameters related to the parameter in the 2nd column of the same row, if applicable.
* The 5th column is the place where the values for the 4th column (if applicable) is printed.