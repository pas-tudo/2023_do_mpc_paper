
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
import cstr_nn_model

show_animation = True


model = cstr_model.get_model()
bound_dict = json.load(open(os.path.join(example_path, 'config','cstr_bounds.json')))
simulator = cstr_simulator.get_simulator(model)

onnx_model = onnx.load(os.path.join('.', 'models', 'cstr_onnx.onnx'))
nn_model = cstr_nn_model.get_nn_model(onnx_model)

mpc, mpc_tvp_template  = cstr_controller.get_mpc(nn_model, bound_dict)

# Set the initial state of mpc and simulator:
C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
C_b_0 = 0.5 # This is the controlled variable [mol/l]
T_R_0 = 134.14 #[C]
T_K_0 = 130.0 #[C]
x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)

mpc.x0 = x0
simulator.x0 = x0

mpc.set_initial_guess()

fig, ax, graphics = cstr_helper.plot_cstr_results_new(mpc.data)


plt.ion()

def update_plot_callback(k):
    graphics.plot_results(t_ind=k)
    graphics.plot_predictions(t_ind=k)
    graphics.reset_axes()
    plt.show()
    plt.pause(0.01)

# Main loop
for k in range(100):
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)
    if show_animation:
        update_plot_callback(k)

input('Press any key to exit.')