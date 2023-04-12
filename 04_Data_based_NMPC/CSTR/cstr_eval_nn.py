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

# CSTR files
example_path = os.path.join('..','..','01_Example_Systems','CSTR')
sys.path.append(example_path)
import cstr_model
import cstr_controller
import cstr_simulator
import cstr_helper


# %% [markdown]
# ## Get do-mpc model from ONNX model
# - Initialize a `do_mpc` model
#     - Introduce states, inputs, and TVP as for the physical model
#     - the model is of type `discrete` because the ONNX model is a discrete-time model
# - Use the ONNX model with `ONNXConversion` to get the symbolic expressions for the next state
#     - Initialize the `ONNXConversion` class with the ONNX model
#     - Call the `convert` method with the symbolic variables as arguments
#     - The `ONNXConversion` class can be indexed to retrieve the symbolic expressions for the next state


# %%
def get_nn_model(onnx):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type, 'SX')


    # States struct (optimization variables):
    C_a = model.set_variable(var_type='_x', var_name='C_a', shape=(1,1))
    C_b = model.set_variable(var_type='_x', var_name='C_b', shape=(1,1))
    T_R = model.set_variable(var_type='_x', var_name='T_R', shape=(1,1))
    T_K = model.set_variable(var_type='_x', var_name='T_K', shape=(1,1))
    # Concatenate states:
    x = vertcat(C_a, C_b, T_R, T_K)

    # TVP
    C_b_set = model.set_variable(var_type='_tvp', var_name='C_b_set', shape=(1,1))

    # Introduce parameters (not used) to comply with physical model
    alpha = model.set_variable(var_type='_p', var_name='alpha', shape=(1,1))
    beta = model.set_variable(var_type='_p', var_name='beta', shape=(1,1))

    # Input struct (optimization variables):
    F = model.set_variable(var_type='_u', var_name='F')
    Q_dot = model.set_variable(var_type='_u', var_name='Q_dot')
    # Concatenate inputs:
    u = vertcat(F, Q_dot)

    # Use ONNX model with converter
    casadi_converter = do_mpc.sysid.ONNXConversion(onnx)
    casadi_converter.convert(x_k = x.T, u_k = u.T)
    x_next = casadi_converter['x_next']
    

    # Differential equations
    model.set_rhs('C_a', x_next[0])
    model.set_rhs('C_b', x_next[1])
    model.set_rhs('T_R', x_next[2]) 
    model.set_rhs('T_K', x_next[3])

    # Build the model
    model.setup()
    return model

# %% [markdown]
# ## Load trained model from ONNX and get do-mpc model
# - Load the ONNX model
# - Get the do-mpc model from the ONNX model
# - Get the simulator for the do-mpc model
# - Get the simulator for the physical model


# %%
def random_x0(bound_dict, key=None):
    return np.array([
        np.random.uniform(lb, ub) for lb, ub in zip(bound_dict['states']['lower'].values(), bound_dict['states']['upper'].values())
    ])

if __name__ == '__main__':
    onnx_model = onnx.load(os.path.join('.', 'models', 'cstr_onnx.onnx'))

    nn_model = get_nn_model(onnx_model)
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

    true_simulator.x0 = random_x0(bound_dict)
    nn_simulator.x0 = copy.copy(true_simulator.x0)

    for k in range(50):
        u0 = random_input.make_step(true_simulator.x0)
        true_simulator.make_step(u0)
        nn_simulator.make_step(u0)

    # %% [markdown]
    # ### Plot the results
    # - Plot the states and inputs of both models
    # %%
    fig, ax = plt.subplots(6, 1, sharex=True)

    cstr_helper.plot_cstr_results(true_simulator.data, (fig, ax), label='Physical model', marker='o', markevery=5, markersize=5, markerfacecolor='none')
    cstr_helper.plot_cstr_results(nn_simulator.data, (fig, ax), label='NN', marker='x', markevery=5, markersize=5, markerfacecolor='none')


    ax[0].legend()
    # %% [markdown]
    # ## Test MPC with NN model

    # %%
    nn_mpc, nn_mpc_template = cstr_controller.get_mpc(nn_model, bound_dict)

    # %% 

    nn_mpc.reset_history()
    true_simulator.reset_history()

    x0 = random_x0(bound_dict).reshape(-1,1)

    true_simulator.x0 = x0
    nn_mpc.x0 = x0 
    nn_mpc.set_initial_guess()

    for k in range(50):
        u0 = nn_mpc.make_step(x0)
        x0 = true_simulator.make_step(u0)

    # %%
    fig, ax = plt.subplots(6, 1, sharex=True)
    cstr_helper.plot_cstr_results(true_simulator.data, (fig, ax), label='Physical model', marker='o', markevery=5, markersize=5, markerfacecolor='none')

# %%
