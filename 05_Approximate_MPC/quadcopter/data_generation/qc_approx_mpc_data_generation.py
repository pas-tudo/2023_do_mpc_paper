
# %% [markdown]
"""
# Data generation for Quadcopter example

Import the necessary packages.
"""
# %% [code]
# Import packages

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import pandas as pd
import time as time
import importlib
import sys 
import os
import json
import pathlib
import multiprocessing as mp
import importlib

from casadi import *
from casadi.tools import *

# CSTR files
example_path = os.path.join('..','..','..','01_Example_Systems','quadcopter')
sys.path.append(example_path)
import qcmodel
import qccontrol
import plot_results
import qctrajectory

# Control packages
import do_mpc

IS_INTERACTIVE = hasattr(sys, 'ps1')

# %% [markdown]

# ## Create sampling plan

# We define $N$ sampling cases with random initial conditions and setpoints for the CSTR. 
# - The initial conditions are sampled uniformly from the bounds defined in the `cstr_bounds.json` file.
# - The setpoint is sampled uniformly from the bounds defined in the `cstr_bounds.json` file.
#
# An impression of the sampling cases is shown below:

# %% [code]

def get_uniform_func(lb, ub):
    def f():
        return np.random.uniform(lb, ub)

    return f


if __name__ ==  '__main__' :
    sp = do_mpc.sampling.SamplingPlanner(overwrite=True)
    data_dir = os.path.join('.', 'closed_loop_mpc')
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    sp.set_sampling_var('p0',   get_uniform_func(-1.5, 1.5))
    sp.set_sampling_var('p1',   get_uniform_func(-1.5, 1.5))
    sp.set_sampling_var('p2',   get_uniform_func(0, 2))
    sp.set_sampling_var('phi0',  get_uniform_func(-np.pi, np.pi))
    sp.set_sampling_var('phi1',  get_uniform_func(-np.pi, np.pi))
    sp.set_sampling_var('phi2',  get_uniform_func(-np.pi, np.pi))
    sp.set_sampling_var('speed',  get_uniform_func(0.2, 1.5))
    sp.set_sampling_var('radius',  get_uniform_func(0.2, 1.5))
    sp.set_sampling_var('height',   get_uniform_func(0.2, 2))
    sp.set_sampling_var('wobble_height',   get_uniform_func(0, 5))
    sp.set_sampling_var('rot', get_uniform_func(-np.pi, np.pi))
    sp.set_sampling_var('input_noise_dist', get_uniform_func(0.5e-3, 2e-3))

    plan = sp.gen_sampling_plan(100)
    sp.export(os.path.join(data_dir, 'sampling_plan_mpc'))



# %% [markdown]
# ## Create sampling function
# 
# We define a sampling function that takes the initial conditions and setpoint as input and returns the closed-loop simulation data.
# Notice that the sampling function must have the same **keyword arguments** as defined in the sampling plan, 
# that is, ``C_a_0``, ``C_b_0``, ``T_R_0``, ``T_K_0``, and ``C_b_set``. The order of these arguments does not matter.
# In order to run the closed-loop simulation, we also need to initialize the following objects:
# - The model
# - The simulator
# - The controller


# %% [code]
t_step = 0.05
qcconf = qcmodel.QuadcopterConfig()
simulator, sim_p_template = qccontrol.get_simulator(t_step, qcmodel.get_model(qcconf, with_pos=True))
mpc, mpc_p_template = qccontrol.get_MPC(t_step, qcmodel.get_model(qcconf, with_pos=True))

def sampling_function(
        p0, p1, p2, phi0, phi1, phi2, speed, radius, height, wobble_height, rot, input_noise_dist
    ):
    simulator.reset_history()
    mpc.reset_history()

    simulator.x0['pos', 0] = p0
    simulator.x0['pos', 1] = p1
    simulator.x0['pos', 2] = p2
    simulator.x0['phi', 0] = phi0
    simulator.x0['phi', 1] = phi1
    simulator.x0['phi', 2] = phi2

    figure_eight_trajectory = qctrajectory.get_wobbly_figure_eight(
        s=speed, 
        a=radius, 
        height=height, 
        wobble=wobble_height, 
        rot=rot, 
        )


    x0 = simulator.x0.cat.full()

    mpc.x0 = x0

    qccontrol.mpc_fly_trajectory(
        simulator, 
        mpc, 
        mpc_p_template, 
        sim_p_template, 
        N_iter=100, 
        trajectory=figure_eight_trajectory,
        noise_dist=input_noise_dist
        )

    return {
        'x_k': mpc.data['_x'], 
        'u_k': mpc.data['_u'], 
        'p_k': simulator.data['_p'], 
        'success': mpc.data['success'], 
        'pos_k': simulator.data['_x', 'pos'],
        }
    
# %% [markdown]
"""
## Sample data

We initialize the ``do_mpc.sampling.Sampler`` object and call the ``sample_data`` method to generate the data.
To initialize the ``Sampler``, we pass:
- The sampling plan
- The sampling function
- A directory where the data will be stored

"""
#%% [code]
if __name__ ==  '__main__' :
    
    sampler = do_mpc.sampling.Sampler(plan, overwrite=True)
    sampler.data_dir = os.path.join('.', 'closed_loop_mpc', '')

    sampler.set_sample_function(sampling_function)

    if IS_INTERACTIVE:
        sampler.sample_idx(0)
    else:
        with mp.Pool(processes=8) as pool:
            p = pool.map(sampler.sample_idx, list(range(sampler.n_samples)))
# %%
