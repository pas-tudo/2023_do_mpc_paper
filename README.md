# do-mpc: Towards FAIR nonlinear and robust model predictive control

Dear visitor,

this is the accompanying repository for our work "do-mpc: Towards FAIR nonlinear and robust model predictive control". In the spirit of FAIR research, the results shown in the paper can be recreated, reused or extended with the materials presented in this repository. Please note that it is required to install [do-mpc](https://www.do-mpc.com/en/latest/) >v4.5.6.

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
As a snapshot of the obtained results, we export the [Python interactive window](https://code.visualstudio.com/docs/python/jupyter-support-py) (supported by Visual Studio Code) as Jupyter Notebooks. These Jupyter Notebooks are saved synchronized with [Git LFS](https://git-lfs.com). **For convenience, we highlight below, which files are Jupyter Notebooks and can be conveniently displayed in your browser.**
Notice that snapshots are not possible for all files (in particular those that require **multiprocessing**). 

### Where to find the investigated models

- The CSTR model introduced in **Section 4** can be found [here](01_Example_Systems/CSTR/cstr_model.py). The model is used in investigations shown in Figure 3 and Figure 6.
- The quadcopter model introduced in **Section 6** can be found [here](01_Example_Systems/quadcopter/qcmodel.py). A demonstration of the model is shown in [this Jupyter Notebook](01_Example_Systems/quadcopter/quadcopter_demo.ipynb). Results obtained with this model are shown in Figure 8.

### Where to find the results

#### Figure 3: Comparison of robust vs. nominal MPC for the CSTR
- [**Jupyter**] The obtained results are created and shown [here](03_LQR2NMPC/CSTR_wo_OPCUA/cstr_comparison_robust_nominal.ipynb).
- [**Jupyter**] Software-in-the-loop simulation with OPC UA (reported results in text) is evaluated [here](03_LQR2NMPC/CSTR_w_OPCUA/evaluation_real_time_vs_ideal.ipynb)

#### Figure 6: MPC controller for CSTR with exact model and neural network model
- System identification data-generation with sampling framework shown [here](04_Data_based_NMPC/CSTR/data_generation/cstr_data_generation.py)
- [**Jupyter**] Training of neural network system model and ONNX export shown [here](https://github.com/pas-tudo/2023_do_mpc_paper/blob/main/04_Data_based_NMPC/CSTR/cstr_train_nn.ipynb)
- Closed-loop simulations using MPC with nominal model and MPC with neural network model for different $p=50$ different cases using the sampling framework shown [here](https://github.com/pas-tudo/2023_do_mpc_paper/blob/main/04_Data_based_NMPC/CSTR/evaluation/cstr_nn_meta_closed_loop_sampling.py).
- [**Jupyter**] Evaluation of results (Figure 6) created [here](https://github.com/pas-tudo/2023_do_mpc_paper/blob/main/04_Data_based_NMPC/CSTR/evaluation/cstr_nn_meta_closed_loop_analysis.ipynb).

#### Figure 8: Closed-loop trajectories for Quadcopter controlled with approximate MPC
- Data-generation for approximate MPC controller using the sampling framework shown [here](05_Approximate_MPC/quadcopter/data_generation/qc_approx_mpc_data_generation.py). This data also contains the sensitivity information which can be obtained with the do-mpc sensitivity module.
- Implementation of Sobolev training for approximate MPC can be found [here](05_Approximate_MPC/quadcopter/qc_train_approx_mpc.py).
- Training of different approximate MPC controllers (varying number of trajectories used in training and with or without Sobolev loss) shown [here](05_Approximate_MPC/quadcopter/meta_analysis/qc_meta_01_train.py).
- [**Jupyter**] Evaluation of approximate MPC controllers for different values of additive input noise shown [here](05_Approximate_MPC/quadcopter/meta_analysis/qc_meta_02_evaluation.ipynb).
