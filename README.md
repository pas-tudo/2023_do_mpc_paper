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

Most results of this work can be recovered by executing the respective Python scripts. A few selected results are also exported as Jupyter Notebooks which can be directly displayed in the browser.
