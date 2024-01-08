# open_strategy_abm
An agent-based model to simulate open strategy dynamics, accompanying the paper "Open Strategy".

## Table of Contents

- [Summary](#summary)
- [Model Workflow](#model-workflow)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Model](#running-the-model)
- [File Structure](#file-structure)
- [Configuration File](#configuration-file)
- [Results](#results)
- [Citation](#citation)

## Summary
This agent-based simulation model explores the impacts of various open strategy-making processes. It demonstrates how an organization can involve stakeholders in different phases of strategy-making: 1) Idea Generation, 2) Strategy Selection, and 3) Implementation. The model evaluates the effects of these participatory approaches on strategy performance and stakeholder engagement.

Key Findings:
- Opening the Idea Generation phase typically leads to superior strategy performance but might result in stakeholder disengagement or performance decline over time.
- Involving stakeholders in both Idea Generation and Strategy Selection can yield varied outcomes, enhancing or diminishing performance compared to closed strategies.

Agent Types:
- **The Organization**: Assumes different roles based on the strategy-making method.
- **Stakeholders**: Their roles vary according to the strategy-making approach.
- **Environment**: Represents environmental changes, acting randomly.

## Model Workflow
[Strategy as a Practice](https://github.com/alibanihashemi/open_strategy_abm/issues/1#issue-2069867436) outlines the general workflow of the model.

## Requirements
Test Environment: Windows 10, Python 3.8, 16GB RAM, Intel Core i7 CPU. Install the latest version of Anaconda for Python 3 and scientific package support.

Required Packages:
- numpy
- matplotlib
- pandas
- scipy
- NKpack
- multiprocessing
- seaborn

Detailed in `requirements.txt`.

## Installation

### Without Docker
1. Download and install [Anaconda (Individual Edition)](https://www.anaconda.com/products/individual-d).
2. Create a virtual environment: `conda create -n myenv python=3.8`.
3. Activate the virtual environment: `conda activate myenv`.
4. Install packages: `pip install -r requirements.txt`.
5. For **surprise** package issues on Windows: `conda install -c conda-forge scikit-surprise`.

### With Docker
Pull the Docker image: `docker pull nadadocker/simulation`.

## Running the Model
For non-Docker setups:

Navigate to the source directory: `cd src`.
Execute `python setup.py`.

## File Structure
Built using [Mesa](https://github.com/projectmesa/mesa), a Python agent-based simulation framework.




## File structure
The simulation is built with the help of [Mesa](https://github.com/projectmesa/mesa), an agent-based simulation framework in Python.
```
├── data/

├── Dockerfile
├── figures/                      <- Figures that show simulation results
│   ├── modelgeneralflow.png
│   ├── modeldescription.png

├── README.md
├── requirements.txt
├── src/
  ├── __init__.py
  ├── simulations.xlsx                  <- Simulation settings
  ├── switch.py                 <- Contains all classes and methods
  ├── setup.py                      <- Launches the simulation
  ├── simulation.txt         <- How to run the simulation
```


## Configuration file
`simulations.xlsx` includes all the required parameters to set up the model.


**Note**: Running the code may take a long time (e.g. one day to one week) based on the predefined time steps and the number of participants and etc in the configuration. 


## Results
Each execution of the model generates a unique folder inside the results folder. The collected data from the simulation contains various CSV files, a summary of the simulated strategies in a file named scenarios.csv, and plots in the PNG and pdf format.


The following is part of the results generated from running the simulation for ...



