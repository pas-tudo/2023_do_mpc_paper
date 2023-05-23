# 2022_do_mpc_paper

Dear visitor,

this is the accompanying repository for our work "do-mpc: Towards FAIR nonlinear and robust model predictive control". In the spirit of FAIR research, the results shown in the paper can be recreated, reused or extended with the materials presented in this repository. Please note that it is required to install [do-mpc]([www.do-mpc.com](https://www.do-mpc.com/en/latest/) >v4.5.6.

## Installing the required packages

We advise to clone this repository and create a local anaconda environment with the required packages. Assuming [miniconda or anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) are installed on the system, use the terminal from the root of this repository and execute:

```
conda env create --prefix ./.conda -f .conda_environment.yml  
```

This will create a directory called ``.conda`` containing the required packages for this repository. Activate the environment with:

```
conda activate ./.conda
```

## Structure of this repository

### Introduction
All results in this repository are created in python. We only have ``.py`` files under source-control (in Github). 
As a snapshot of the obtained results, we export the [Python interactive window](https://code.visualstudio.com/docs/python/jupyter-support-py) (supported by Visual Studio Code) as Jupyter Notebooks. These Jupyter Notebooks are saved synchronized with [Git LFS](https://git-lfs.com).
Notice that snapshots are not possible for all files (in particular those that require **multiprocessing**). 

### Where to find the results
