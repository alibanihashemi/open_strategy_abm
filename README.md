# open_strategy_abm
An agent based model to simulate open strategy
Online material for the paper "Open Strategy"

## Table of content

- [Summary](#summary)
- [General model workflow](#general-model-workflow)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the model](#running-the-model)
- [File structure](#file-structure)
- [Rating dataset](#ratings-dataset)
- [Configuration file](#configuration-file)
- [Results](#results)



## Summary
This agent-based simulation model demonstrates the consequences of various methods of stratgy making. An organization can open up its strategy in different levels to stakeholders. There are three phases of strategy making: 1-Idea generation 2-strategy selection 3-Implementation. Organization can keep the process of strategy making closed or include stakeholders in one or two first phases of strategy making.
When first phase of strategy making is opened the performance of strategy is the best however there might be diengagement of stakeholders or even lose of performance through time.
When the idea generation phase and strategy selection phase are both open the performance can be better or worse as compared to close strategy making. Nevertheless stakeholders are better engaged in the process of strategy making.

Three types of agents are used in the model: 

<ul>
<li> The organization: The organization has deferent roles depend on the strategy making method
<ul>
<li> It proposes strategies in idea generation phase </li>
<li> It selects a strategy in the second phse </li>
<li> It implements the selected stratgy</li>
</ul>
</li>

<li> Stakeholders: Stakeholders have different roles based on the method of strategy making
<ul>
<li> They propose strategies in idea generation phase </li>
<li> They vote on strategies in strategy selection phase</li>
</ul>
</li>

<li>Environment: This agent represents the environmental changes and acts random </li>
</ul>

## General model workflow based on [Startegy as a practice ](https://github.com/alibanihashemi/open_strategy_abm/issues/1#issue-2069867436)

## Requirements
We tested the code on a machine with MS Windows 10, Python=3.8, 16GB, and an Intel Core 7 CPU.  Install the last version of Anaconda, which comes with Python 3 and supports scientific packages.

The following packages are used in our model, see also the file `requirements.txt`:
* [numpy](https://numpy.org/)
* [matplotlib](https://matplotlib.org/)
* [pandas](https://pandas.pydata.org/)
* [scipy](https://www.scipy.org/)
* [NKpack](https://pypi.org/project/nkpack/)
* [multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
* [seaborn](https://seaborn.pydata.org/)

## Installation

### Setting up the environment (No Docker)
Download and install [Anaconda](https://www.anaconda.com/products/individual-d) (Individual Edition)

Create a virtual environment
```
conda create -n myenv python=3.8
```
Activate the virtual environment 
```
conda activate myenv
```
More commands regarding the use of virtual environments in Anaconda can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) 

Install the required packages by running: 
```
pip install -r requirements.txt
```

If you face errors when insatlling the **surprise** package on MS Windows, run:
```
conda install -c conda-forge scikit-surprise
```
### Setting up the environment (Using Docker)
We provide a Docker image on Docker hub; to pull the image use the following:

```
docker pull nadadocker/simulation
```

## Running the model
To run the simulation when Docker does not exist: 

```
cd src
```

```python setup.py```


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



