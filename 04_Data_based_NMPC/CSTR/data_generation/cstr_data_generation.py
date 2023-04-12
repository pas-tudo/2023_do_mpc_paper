# %% [markdown]
"""
# Data generation for CSTR example

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
example_path = os.path.join('..','..','..','01_Example_Systems','CSTR')
sys.path.append(example_path)
import cstr_model
import cstr_controller
import cstr_simulator
from cstr_helper import get_random_uniform_func

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

bound_dict = json.load(open(os.path.join(example_path, 'config','cstr_bounds.json')))



if __name__ ==  '__main__' :
    sp = do_mpc.sampling.SamplingPlanner(overwrite=True)
    data_dir = os.path.join('.', 'closed_loop_lqr')
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)



    sp.set_sampling_var('C_a_0',   get_random_uniform_func(bound_dict, 'states', 'C_a'))
    sp.set_sampling_var('C_b_0',   get_random_uniform_func(bound_dict, 'states', 'C_b'))
    sp.set_sampling_var('T_R_0',   get_random_uniform_func(bound_dict, 'states', 'T_R'))
    sp.set_sampling_var('T_K_0',   get_random_uniform_func(bound_dict, 'states', 'T_K'))
    sp.set_sampling_var('C_b_set', get_random_uniform_func(bound_dict, 'states', 'C_b'))
    sp.set_sampling_var('random_contribution', np.random.rand)

    plan = sp.gen_sampling_plan(100)
    sp.export(os.path.join(data_dir, 'sampling_plan_lqr'))




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
model = cstr_model.get_model()
simulator = cstr_simulator.get_simulator(model)
lqr_clipper = cstr_controller.get_clipper(model, bound_dict)
random_input = cstr_controller.UniformRandomInput(bound_dict)

def sampling_function(
        C_a_0, C_b_0, T_R_0, T_K_0, C_b_set, random_contribution
    ):
    x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)

    linear_model, xss, uss = cstr_model.get_linear_model(C_b_set = C_b_set)
    lqr = cstr_controller.get_lqr(linear_model, xss, uss)

    simulator.reset_history()

    simulator.x0 = x0
    lqr.x0 = x0

    for k in range(500):
        u0 = (1-random_contribution)*lqr.make_step(x0) + random_contribution*random_input.make_step(x0)
        u0 = lqr_clipper(u0) 
        x0 = simulator.make_step(u0)

    return simulator.data
    
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
    sampler.data_dir = os.path.join('.', 'closed_loop_lqr', '')

    sampler.set_sample_function(sampling_function)

    if IS_INTERACTIVE:
        sampler.sample_data()
    else:
        with mp.Pool(processes=4) as pool:
            p = pool.map(sampler.sample_idx, list(range(sampler.n_samples)))
# %% [markdown]
# # First impression of sampling results
# Load a ``do_mp.sampling.DataHandler`` object to analyze the sampling results.
#
# We plot the family of trajectories to get an impression.
# %%
if __name__ == '__main__':
    dh = do_mpc.sampling.DataHandler(plan)
    dh.data_dir = sampler.data_dir

    fig, ax = plt.subplots(6,1, sharex=True)

    for data_i in dh[:]:
        ax[0].plot(data_i['res']['_time'], data_i['res']['_x','C_a'], color='k', alpha=.1, linewidth=1)
        ax[1].plot(data_i['res']['_time'], data_i['res']['_x','C_b'], color='k', alpha=.1, linewidth=1)
        ax[2].plot(data_i['res']['_time'], data_i['res']['_x','T_R'], color='k', alpha=.1, linewidth=1)
        ax[3].plot(data_i['res']['_time'], data_i['res']['_x','T_K'], color='k', alpha=.1, linewidth=1)
        ax[4].step(data_i['res']['_time'], data_i['res']['_u','F'], color='k', alpha=.1, linewidth=1, where='post')
        ax[5].step(data_i['res']['_time'], data_i['res']['_u','Q_dot'], color='k', alpha=.1, linewidth=1, where='post')

    ax[0].set_ylabel('C_a')
    ax[1].set_ylabel('C_b')
    ax[2].set_ylabel('T_R')
    ax[3].set_ylabel('T_K')
    ax[4].set_ylabel('F')
    ax[5].set_ylabel('Q_dot')
    ax[5].set_xlabel('Time [h]')

    fig.align_ylabels()
    


# %%
