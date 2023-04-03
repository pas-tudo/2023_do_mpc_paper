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
def setup_nominal_mpc():
    n_scenarios = 1
    mpc, mpc_tvp_template, mpc_p_template = CSTRcontrol.get_MPC(
        t_step=1, CSTR=cstr, n_robust=0, n_scenarios=n_scenarios, store_full_solution=True)

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

    return mpc


def plot_pred(mpc):
    graphics = do_mpc.graphics.Graphics(mpc.data)

    fig, ax = plt.subplots(5, sharex=True)

    graphics.add_line(var_type='_x', var_name='cA', axis=ax[0], alpha=.5)
    graphics.add_line(var_type='_x', var_name='cB', axis=ax[1], alpha=.5)
    graphics.add_line(var_type='_x', var_name='cR', axis=ax[2], alpha=.5)
    graphics.add_line(var_type='_x', var_name='cS', axis=ax[3], alpha=.5)
    graphics.add_line(var_type='_x', var_name='Tr', axis=ax[4], alpha=.5)

    ax[0].set_ylabel('cA')
    ax[1].set_ylabel('cB')
    ax[2].set_ylabel('cR')
    ax[3].set_ylabel('cS')
    ax[4].set_ylabel('Tr')
    ax[4].set_xlabel('Time [h]')

    graphics.plot_predictions()
    for ax_i in ax:
        ax_i.relim()
        ax_i.autoscale()

    return fig, ax, graphics

# %% [markdown]
"""
### Closed-loop simulation with nominal MPC
"""
# %% [code]
if __name__ == '__main__':
    mpc = setup_nominal_mpc()

    mpc.reset_history()
    simulator.reset_history()

    mpc.x0['Tr'] = 20
    simulator.x0['Tr'] = 20

    mpc.set_initial_guess()

    # %% [code]
    # Plot initial prediction
    mpc.make_step(mpc.x0)
    fig, ax, graphics = plot_pred(mpc)

    # %%
    mpc.data.prediction(('_x', 'cR', -1),-1)[0,-1, 0]


    # %% [code]

    # Closed-loop simulation
    CSTRcontrol.run_closed_loop(mpc, simulator, n_steps = 35)
    
    # %% [code]
    do_mpc.graphics.default_plot(simulator.data, figsize=(10, 10))

# %% [markdown]
"""
## Robust MPC

### Preparation of scenarios for the robust MPC
- Create all possible combinations of the parameters ``k1_var``, ``k2_var``, ``delH1_var`` and ``delH2_var``.
- The number of scenarios is the number of combinations.
- Create new instance of the mpc controller with ``n_scenarios`` scenarios
- Set the ``mpc_p_template`` to the scenarios
"""
# %%

def setup_robust_mpc():

    k1_var = [0.7, 1.3]
    k2_var = [0.7, 1.3]
    delH1_var = [0.7, 1.3]
    delH2_var = [0.7, 1.3]

    scenarios = list(itertools.product(k1_var, k2_var, delH1_var, delH2_var))
    n_scenarios = len(scenarios)

    mpc, mpc_tvp_template, mpc_p_template = CSTRcontrol.get_MPC(
        t_step=1, CSTR=cstr, n_robust=1, n_scenarios=n_scenarios, store_full_solution=True)


    mpc_tvp_template['_tvp', :, 'k1_mean'] = cstr.k1_0
    mpc_tvp_template['_tvp', :, 'k2_mean'] = cstr.k2_0
    mpc_tvp_template['_tvp', :, 'delH1_mean'] = cstr.delH1_0
    mpc_tvp_template['_tvp', :, 'delH2_mean'] = cstr.delH2_0

    for i, scenario_i in enumerate(scenarios):
        mpc_p_template['_p', i, 'k1_var'] = scenario_i
        mpc_p_template['_p', i, 'k2_var'] = scenario_i
        mpc_p_template['_p', i, 'delH1_var'] = scenario_i
        mpc_p_template['_p', i, 'delH2_var'] = scenario_i

    return mpc

# %% [markdown]
"""
### Compute open-loop solution for the robust MPC
"""

# %%
if __name__ == '__main__':
    mpc = setup_robust_mpc()

   
    mpc.reset_history()
    simulator.reset_history()

    mpc.x0['Tr'] = 20
    simulator.x0['Tr'] = 20

    mpc.set_initial_guess()

    # %%

    mpc.make_step(mpc.x0)

    # %%
    fig, ax, graphics = prepare_pred_plot(mpc)


# %%
