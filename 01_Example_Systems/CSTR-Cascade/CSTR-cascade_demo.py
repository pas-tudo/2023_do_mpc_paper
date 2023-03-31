# %% [markdown]
"""
# Demo file for the CSTR cascade example.

Import the necessary packages.
"""
# %% [code]
# Essentials
import CSTRcontrol
import CSTRmodel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Tools
from IPython.display import clear_output
import copy
import pandas as pd
import time as time
import importlib
import itertools

# Specialized packages
from casadi import *
from casadi.tools import *

# Plotting
from matplotlib.animation import FuncAnimation, ImageMagickFileWriter
import time as time

# Control packages
import do_mpc

# %% [markdown]
"""
## Get the CSTR model, simulator and controller

The model can be adapted by changing the number of reactors in the cascade.
"""
# %% [code]
importlib.reload(CSTRmodel)
importlib.reload(CSTRcontrol)

cstr = CSTRmodel.CSTR_Cascade(n_reac=4)


cstr.get_model()
simulator, sim_tvp_template, sim_p_template = CSTRcontrol.get_simulator(
    t_step=1, CSTR=cstr)


n_scenarios = 1

mpc, mpc_tvp_template, mpc_p_template = CSTRcontrol.get_MPC(
    t_step=1, CSTR=cstr, n_robust=0, n_scenarios=n_scenarios)


# %% [markdown]
"""
## Nominal MPC


Set parameters and time-varying parameters for the Simulator and the MPC controller.

- For the simulator and MPC controller use the same values for ``k1_mean`` and ``k2_mean``.
The true value of ``k1`` and ``k2`` are modified in the simulator with the ``k1_var`` and ``k2_var`` parameters.
A similar modiifcation is done to the parameters ``delH1`` and ``delH2``.
- The **nominal MPC** controller has no knowledge of the uncertainty in the parameters and 
is setup with ``k1_var``, ``k2_var``, ``delH1_var`` and ``delH2_var`` set to one.
- In the first test, the true values of the parameters are used for the simulator, 
that is, ``k1_var``, ``k2_var``, ``delH1_var`` and ``delH2_var`` are set to one.

"""
# %% [code]
# Simulator
sim_tvp_template['k1_mean'] = cstr.k1_0
sim_tvp_template['k2_mean'] = cstr.k2_0
sim_p_template['k1_var'] = 1
sim_p_template['k2_var'] = 1

sim_tvp_template['delH1_mean'] = cstr.delH1_0
sim_tvp_template['delH2_mean'] = cstr.delH2_0
sim_p_template['delH1_var'] = 1
sim_p_template['delH2_var'] = 1

# MPC
mpc_tvp_template['_tvp', :, 'k1_mean'] = cstr.k1_0
mpc_tvp_template['_tvp', :, 'k2_mean'] = cstr.k2_0
mpc_tvp_template['_tvp', :, 'delH1_mean'] = cstr.delH1_0
mpc_tvp_template['_tvp', :, 'delH2_mean'] = cstr.delH2_0

mpc_p_template['_p', :, 'k1_var'] = 1
mpc_p_template['_p', :, 'k2_var'] = 1
mpc_p_template['_p', :, 'delH1_var'] = 1
mpc_p_template['_p', :, 'delH2_var'] = 1

# %% [markdown]
"""
### Closed-loop simulation with nominal MPC
"""
# %% [code]
mpc.reset_history()
simulator.reset_history()

mpc.x0['Tr'] = cstr.Tr_in
simulator.x0['Tr'] = cstr.Tr_in

mpc.set_initial_guess()

# Closed-loop simulation
x0 = simulator.x0
for k in range(40):
    u0 = mpc.make_step(x0)
    x0 = DM(simulator.make_step(u0))


# %% [code]
# Plot the results with do-mpc
do_mpc.graphics.default_plot(simulator.data, figsize=(10, 16))
# Default plot
# %%
