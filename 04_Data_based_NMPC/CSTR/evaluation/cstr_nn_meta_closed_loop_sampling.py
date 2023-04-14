# %% [markdown]
"""
# Data generation for CSTR example

Import the necessary packages.
"""
# %% [code]
# Import packages

import numpy as np
import pandas as pd

import pandas as pd
import time as time
import sys 
import os
import json
import pathlib
import multiprocessing as mp
import importlib

from casadi import *
from casadi.tools import *
import onnx

# CSTR files
example_path = os.path.join('..','..','..','01_Example_Systems','CSTR')
sys.path.append(example_path)
sys.path.append('..')
import cstr_model
import cstr_controller
import cstr_simulator
import cstr_helper
from cstr_helper import get_random_uniform_func
import cstr_nn_model


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

# Increase the bounds for the intial temperatures to avoid stalling the reactor.
bound_dict['states']['lower']['T_R'] = 90
bound_dict['states']['lower']['T_K'] = 90


if __name__ ==  '__main__' :
    sp = do_mpc.sampling.SamplingPlanner(overwrite=True)
    data_dir = os.path.join('.', 'closed_loop_meta')
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    sp.set_sampling_var('C_a_0',   get_random_uniform_func(bound_dict, 'states', 'C_a', reduce_range = 0.3))
    sp.set_sampling_var('C_b_0',   get_random_uniform_func(bound_dict, 'states', 'C_b', reduce_range = 0.3))
    sp.set_sampling_var('T_R_0',   get_random_uniform_func(bound_dict, 'states', 'T_R', reduce_range = 0.3))
    sp.set_sampling_var('T_K_0',   get_random_uniform_func(bound_dict, 'states', 'T_K', reduce_range = 0.3))
    sp.set_sampling_var('C_b_set', get_random_uniform_func(bound_dict, 'states', 'C_b', reduce_range = 0.))

    plan = sp.gen_sampling_plan(50)
    sp.export(os.path.join(data_dir, 'sampling_plan_closed_loop_meta'))


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
onnx_model = onnx.load(os.path.join('..', 'models', 'cstr_onnx.onnx'))
nn_model = cstr_nn_model.get_nn_model(onnx_model)
simulator = cstr_simulator.get_simulator(model)
bound_dict = json.load(open(os.path.join(example_path, 'config','cstr_bounds.json')))
mpc_true_model, tvp_true_model = cstr_controller.get_mpc(model, bound_dict, {'store_full_solution': False})
mpc_nn_model, tvp_nn_model = cstr_controller.get_mpc(nn_model, bound_dict, {'store_full_solution': False})


def sampling_function(
        C_a_0, C_b_0, T_R_0, T_K_0, C_b_set 
    ):

    x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)
    tvp_true_model['_tvp',:,'C_b_set'] = C_b_set
    tvp_nn_model['_tvp',:,'C_b_set'] = C_b_set

    simulator.reset_history()
    mpc_true_model.reset_history()
    simulator.x0 = x0
    mpc_true_model.x0 = x0
    mpc_true_model.set_initial_guess()

    # Run MPC with exact model
    for k in range(50):
        u0 = mpc_true_model.make_step(x0)
        x0 = simulator.make_step(u0)

    x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)
    tvp_true_model['_tvp',:,'C_b_set'] = C_b_set
    tvp_nn_model['_tvp',:,'C_b_set'] = C_b_set

    simulator.reset_history()
    mpc_nn_model.reset_history()
    simulator.x0 = x0
    mpc_nn_model.x0 = x0
    mpc_nn_model.set_initial_guess()

    # Run MPC with exact model
    for k in range(50):
        u0 = mpc_nn_model.make_step(x0)
        x0 = simulator.make_step(u0)

    res = {
        'mpc_true': mpc_true_model.data,
        'mpc_nn': mpc_nn_model.data,
    }

    return res 
    
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
    sampler.data_dir = os.path.join(data_dir, '')

    sampler.set_sample_function(sampling_function)

    if IS_INTERACTIVE:
        sampler.sample_idx(0)
    else:
        with mp.Pool(processes=6) as pool:
            p = pool.map(sampler.sample_idx, list(range(sampler.n_samples)))



# %%
if False:

    dh = do_mpc.sampling.DataHandler(plan)
    dh.data_dir = sampler.data_dir

    dh[0]
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(4, 1, sharex=True)
    

    _,_, sim_graphics = cstr_helper.plot_cstr_results_new(dh[0][0]['res']['mpc_true'], (fig, ax), marker='o', markevery=5, markersize=5, linewidth = 3, alpha = .5, markerfacecolor='none', with_legend=False, with_setpoint=False)
    _ = [ax_i.set_prop_cycle(None) for ax_i in ax]
    c_,_, nn_graphics = cstr_helper.plot_cstr_results_new(dh[0][0]['res']['mpc_nn'], (fig, ax), marker='x', markevery=5, markersize=5, markerfacecolor='none', with_legend=False, with_setpoint=False)

    sim_graphics.result_lines['_x','C_a'][0].set_label('True')
    nn_graphics.result_lines['_x','C_a'][0].set_label('NN')
    sim_graphics.plot_predictions(-1)
    nn_graphics.plot_predictions(-1)

    sim_graphics.reset_axes()
# %%
