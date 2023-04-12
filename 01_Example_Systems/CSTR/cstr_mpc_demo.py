#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
import json
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc


import matplotlib.pyplot as plt
import pickle
import time

import cstr_model
import cstr_controller
import cstr_simulator
import cstr_helper

""" User settings: """
show_animation = True
store_results = False


model = cstr_model.get_model()
bound_dict = json.load(open(os.path.join('config','cstr_bounds.json')))
mpc, mpc_tvp_template  = cstr_controller.get_mpc(model, bound_dict)
simulator = cstr_simulator.get_simulator(model)

# Set the initial state of mpc and simulator:
C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
C_b_0 = 0.5 # This is the controlled variable [mol/l]
T_R_0 = 134.14 #[C]
T_K_0 = 130.0 #[C]
x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)

mpc.x0 = x0
simulator.x0 = x0

mpc.set_initial_guess()


# Initialize graphic:
graphics = do_mpc.graphics.Graphics(mpc.data)
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
