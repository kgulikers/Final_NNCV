# Final Assignment: Cityscapes Challenge

**Author:** Kevin Gulikers  
**TU/e Email:** k.h.f.gulikers@student.tue.nl  
**Codalab Username:** Kevke1111



## Overview
This repository contains the final assignment for the Cityscapes Challenge. It includes all code, configurations, and instructions necessary to train the model for semantic segmentationon the Cityscapes dataset.



## Getting Started

### Clone the Repository
```
git clone https://github.com/kgulikers/Final_NNCV.git
cd Final_NNCV
```

### Setup SLURM

For setting up the SLURM cluster, the `README-Slurmd.md` file is  added in this repository. 


### Download the Dataset
Within the running container, execute:

```bash
bash docker/download_data.sh
```

This will download and unpack the Cityscapes dataset into `data/`.

To install the required libraries necessary for training, simply run the following command in your terminal:

```
pip install -r requirements.txt
```

## Running the Code


### SLURM Cluster Execution
Running via SLURM is explained in `README-Slurm.md`.


Make sure to set up WandB paths and API keys in the .env file.

