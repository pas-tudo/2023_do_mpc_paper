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
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
from do_mpc.tools import Timer

import matplotlib.pyplot as plt
import pickle
import time

import cstr_model
import cstr_lqr
import cstr_simulator

""" User settings: """
show_animation = True
store_results = False


model = cstr_model.get_model()
linear_model, xss, uss = cstr_model.get_linear_model()
lqr = cstr_lqr.get_lqr(linear_model, xss, uss)
simulator = cstr_simulator.get_simulator(model)
lqr_clipper = cstr_lqr.get_clipper(linear_model)

# Set the initial state of mpc and simulator:
C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
C_b_0 = 0.5 # This is the controlled variable [mol/l]
T_R_0 = 134.14 #[C]
T_K_0 = 130.0 #[C]
x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)

lqr.x0 = x0
simulator.x0 = x0

for k in range(50):
    u0 = lqr.make_step(x0)
    u0 = lqr_clipper(u0)
    x0 = simulator.make_step(u0)

fig, ax, graphics = do_mpc.graphics.default_plot(simulator.data)
plt.show(block=True)