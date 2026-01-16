# Microcrack Generation Simulation

## Purpose

*(Add a short description here explaining what the simulation does and its intended use.)*

## Scripts

- `Main_Script_pointwiseStress_PlottingAndStatistics_UPDATED_V5_demo.py`
- `input_parameters.py`
- `micro_VoidCrack_pointwiseStress_SimStage1.py`
- `stresses.py`

## How the Scripts Relate

### `Main_Script_pointwiseStress_PlottingAndStatistics_UPDATED_V5_demo.py`

The main executable script.  
This script sets up the simulation, runs it, saves results, performs statistical analysis, and produces plots.

### `input_parameters.py`

Specifies material parameters and dynamic parameters for crack propagation.

### `micro_VoidCrack_pointwiseStress_SimStage1.py`

Defines the `Microcrack` class, which tracks microcrack geometry and implements microcrack behaviour.

### `stresses.py`

Defines functions for calculating stresses in the system using analytical solutions for the stress field near a crack tip.
