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

```python run.py```

When using Docker:
Since the simulation saves data to the disk at the end, an output directory has to be provided to the Docker image. The following command runs a new container of the simulation and saves the output in the "results" directory. Before running the Docker container, create a directory named `results` on the host machine by executing the following commands: 

```
git clone <git_repo>
```

```
cd simulation
```

```
mkdir results
```

Run the Docker container.

```
docker run -dit --rm -v ${PWD}/results:/results --name <my_container> <nadadocker/simulation>
```

* `container_name`: A name of the container
* `${PWD}`: The current working directory
* `-v ${PWD}/results:/results`: Sets up a bind mount volume that links the `/results` directory from inside the 'container_name' to the  directory ${PWD}/results on the host machine. Docker uses  ':' to split the host’s path from the container path, and the host path always comes first
* `<nadadocker/simulation>` : The Docker image that is used to run the container


## File structure
The simulation is built with the help of [Mesa](https://github.com/projectmesa/mesa), an agent-based simulation framework in Python.
```
├── data/
│   ├── dataset                 <- MovieLens dataset 
│   │   ├── movies.csv
│   │   └── ratings.csv
│   ├── recdata/                  <- Recommendation algorithm output saved in  pickle format
│   │   ├── consumers_items_utilities_predictions.p
│   │   ├── consumers_items_utilities_predictions_popular.p
│   │   └── SVDmodel.p
│   └── trust/                    <- Initial data for consumer trust 
│       └── beta_initials.p
├── Dockerfile
├── figures/                      <- Figures that show simulation results
│   ├── modelgeneralflow.png
│   ├── time-consumption_probability.png
│   ├── time-total_profit.png
│   └── time-trust.png
├── README.md
├── requirements.txt
├── results-analysis(R)/                      <- R code to analyze model output, the output is stored in "results" folder, we store it in a seafile service
├── src/
  ├── __init__.py
  ├── config.yml                  <- Simulation settings
  ├── consumer.py                 <- Contains all propoerties and behaviors of consumer agents 
  ├── mesa_utils/
  │   ├── __init__.py
  │   ├── datacollection.py
  │   └── schedule.py
  ├── model.py                    <- Contains the model class, which manages agent creation, data sharing, and simulation output collection 
  ├── plots.py                    <- Plotting module for data analysis
  ├── read_config.py
  ├── run.py                      <- Launches the simulation
  ├── service_provider.py         <- Contains all properties and behavior of the service provider agent
  ├── test.py
  └── utils.py             <- An auxiliary module 

```
## Ratings dataset
We use the [MovieLens dataset](https://grouplens.org/datasets/movielens/), the small version (1 MB), which contains movie ratings for multiple consumers, [more details](http://files.grouplens.org/datasets/movielens/ml-latest-small-README.html). The following shows the content of `ratings.csv`.


|userId|movieId|rating|timestamp|
|------|-------|------|---------|
|1|	1|4|964982703|
|1|3|4|964981247|
|1|6|4|964982224|
|1|47|5|964983815|
|1|50|5|964982931|

The dataset is used to predict consumer items utilities, and to initialize the model.

## Configuration file
`config.yml` includes all the required parameters to set up the model.


**Note**: Running the code may take a long time (e.g. one hour) based on the predefined time steps and the number of replications in the configuration. 


## Results
Each execution of the model generates a unique folder inside the results folder. The collected data from the simulation contains various CSV files, a summary of the simulated strategies in a file named scenarios.json, and plots in the PNG format.


The following is part of the results generated from running the simulation for 1000 time steps and 3 replications. The simulation comprises one service provider and 610 consumers, and consumers can share their experiences on social media.


<table>
  <tr>
    <td>Consumption probability</td>
     <td>Profit per step</td>
     <td>Cumulative profit</td>
  </tr>
  <tr>
    <td><img src="figures/consumption.png" width=300 height=200></td>
    <td><img src="figures/profit-per-step.png" width=300 height=200></td>
    <td><img src="figures/cumulative-profit.png" width=300 height=200></td>
  </tr>
 </table>
