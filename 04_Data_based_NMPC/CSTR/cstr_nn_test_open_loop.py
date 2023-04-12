# %% [markdown]
# # Evaluation of trained neural network model for the CSTR system
# Import packages

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from casadi import *
import os
import sys
import do_mpc
import json
import onnx
import copy
import importlib

# CSTR files
example_path = os.path.join('..','..','01_Example_Systems','CSTR')
sys.path.append(example_path)
import cstr_model
import cstr_controller
import cstr_simulator
import cstr_helper
import cstr_nn_model

# %% [markdown]
# ## Get do-mpc model from ONNX model
# - Initialize a `do_mpc` model
#     - Introduce states, inputs, and TVP as for the physical model
#     - the model is of type `discrete` because the ONNX model is a discrete-time model
# - Use the ONNX model with `ONNXConversion` to get the symbolic expressions for the next state
#     - Initialize the `ONNXConversion` class with the ONNX model
#     - Call the `convert` method with the symbolic variables as arguments
#     - The `ONNXConversion` class can be indexed to retrieve the symbolic expressions for the next state
#
# The code can be found in `cstr_model_nn.py`.


# %% [markdown]
# ## Load trained model from ONNX and get do-mpc model
# - Load the ONNX model
# - Get the do-mpc model from the ONNX model
# - Get the simulator for the do-mpc model
# - Get the simulator for the physical model


# %%
if __name__ == '__main__':
    onnx_model = onnx.load(os.path.join('.', 'models', 'cstr_onnx.onnx'))

    nn_model = cstr_nn_model.get_nn_model(onnx_model)
    true_model = cstr_model.get_model()

    nn_simulator = cstr_simulator.get_simulator(nn_model)
    true_simulator = cstr_simulator.get_simulator(true_model)

    # %% [markdown]  
    # ## Simulate both models and compare the outputs
    # - Initialize the states of both simulators with the same random initial state
    # - Initialize `random_input` class with the bounds of the physical model
    # - Simulate both models for 50 steps with the same random input

    # %%

    bound_dict = json.load(open(os.path.join(example_path, 'config','cstr_bounds.json')))
    random_input = cstr_controller.UniformRandomInput(bound_dict)

    true_simulator.x0 = cstr_helper.get_random_uniform_func(bound_dict, 'states', ['C_a', 'C_b', 'T_R', 'T_K'])()
    nn_simulator.x0 = copy.copy(true_simulator.x0)

    for k in range(50):
        u0 = random_input.make_step(true_simulator.x0)
        true_simulator.make_step(u0)
        nn_simulator.make_step(u0)

    # %% [markdown]
    # ### Plot the results
    # - Plot the states and inputs of both models
    # %%
    fig, ax = plt.subplots(4, 1, sharex=True)
    

    _,_, sim_graphics = cstr_helper.plot_cstr_results_new(true_simulator.data, (fig, ax), marker='o', markevery=5, markersize=5, linewidth = 3, alpha = .5, markerfacecolor='none', with_legend=False, with_setpoint=False)
    _ = [ax_i.set_prop_cycle(None) for ax_i in ax]
    c_,_, nn_graphics = cstr_helper.plot_cstr_results_new(nn_simulator.data, (fig, ax), marker='x', markevery=5, markersize=5, markerfacecolor='none', with_legend=False, with_setpoint=False)

    sim_graphics.result_lines['_x','C_a'][0].set_label('True')
    nn_graphics.result_lines['_x','C_a'][0].set_label('NN')

    ax[0].legend(ncols=3)
    