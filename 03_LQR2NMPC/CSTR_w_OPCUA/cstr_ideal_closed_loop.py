#%% Import libraries
import pickle
import numpy as np
import time as time
import sys 
import os
import json
from casadi import *
from casadi.tools import *

# CSTR files
# example_path = os.path.join('..','..',01_Example_Systems','CSTR')
example_path = os.path.join('01_Example_Systems','CSTR')
sys.path.append(example_path)
import cstr_model
import cstr_controller
import cstr_simulator

#%% Initialize do-mpc classes and define functions for closed loop operation

# Initiate do-mpc classes
model = cstr_model.get_model()
bound_dict = json.load(open(os.path.join(example_path, 'config','cstr_bounds.json')))
mpc, mpc_tvp_template  = cstr_controller.get_mpc(model, bound_dict)
simulator = cstr_simulator.get_simulator(model, t_step=0.000556)


# A function to define an initial state for a do-mpc object
def set_initial_and_reset(obj):
    C_a_0 = 0.2 # This is the initial concentration inside the tank [mol/l]
    C_b_0 = 0.5 # This is the controlled variable [mol/l]
    T_R_0 = 120.0 #[C]
    T_K_0 = 120.0 #[C]
    x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)
    obj.x0 = x0

# A function to run the closed loop
def run_closed_loop(controller, simulator, N_sim = 100):
    # Set the initial state of mpc and simulator:
    set_initial_and_reset(controller)
    set_initial_and_reset(simulator)
    controller.set_initial_guess()

    x0 = simulator.x0
    # Main loop
    for k in range(N_sim):
        u0 = controller.make_step(x0)
        for i in range(int(controller.settings.t_step*3600/(simulator.settings.t_step*3600))):
            x0 = simulator.make_step(u0)

#%% Run the closed loop and save simulator.data as .pkl

run_closed_loop(mpc, simulator, N_sim = 300)

with open('cstr_res_closed_loop_ideal.pkl','wb') as handle:
    pickle.dump(simulator.data, handle, protocol=pickle.HIGHEST_PROTOCOL)