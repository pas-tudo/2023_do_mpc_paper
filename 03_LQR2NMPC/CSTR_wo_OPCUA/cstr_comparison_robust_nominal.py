# %% [markdown]
"""
# Robust multi-stage with CSTR

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
import importlib

from casadi import *
from casadi.tools import *

# CSTR files
example_path = os.path.join('..','..','01_Example_Systems','CSTR')
sys.path.append(example_path)
import cstr_model
import cstr_controller
import cstr_simulator

# Control packages
import do_mpc

IS_INTERACTIVE = hasattr(sys, 'ps1')


plot_path = os.path.join('..','..','00_plotting')
sys.path.append(plot_path)
import mplconfig
importlib.reload(mplconfig)
mplconfig.config_mpl(os.path.join('..','..','00_plotting','notation.tex'))

# %% [markdown]

# # Get do-mpc module instances

# %% [code]


model = cstr_model.get_model()
bound_dict = json.load(open(os.path.join(example_path, 'config','cstr_bounds.json')))
mpc, mpc_tvp_template  = cstr_controller.get_mpc(model, bound_dict)
simulator = cstr_simulator.get_simulator(model)


# %%

def set_initial_and_reset(obj):
    # Reset history
    obj.reset_history()
    
    # Set the initial state of mpc and simulator:
    C_a_0 = 0.2 # This is the initial concentration inside the tank [mol/l]
    C_b_0 = 0.5 # This is the controlled variable [mol/l]
    T_R_0 = 120 #[C]
    T_K_0 = 120.0 #[C]
    x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)

    obj.x0 = x0

def run_closed_loop(controller, simulator, N_sim = 100):
    # Set the initial state of mpc and simulator:
    set_initial_and_reset(controller)
    set_initial_and_reset(simulator)

    controller.set_initial_guess()
    
    # Main loop
    x0 = simulator.x0.cat.full()

    for k in range(N_sim):
        u0 = controller.make_step(x0)
        x0 = simulator.make_step(u0)


def plot_results(controller, simulator, bounds, fig_ax = None):

    if fig_ax is None:
        fig, ax = plt.subplots(3,1, figsize=(mplconfig.columnwidth, 2.3), sharex=True, dpi=200)
    else:
        fig, ax = fig_ax

    t = simulator.data['_time']
    t_pred = controller.t0 + controller.t_step*np.arange(-1, controller.n_horizon).reshape(-1,1)


    # C_a
    line_ca = ax[0].plot(t, simulator.data['_x','C_a'], label='$C_a$')

    if fig_ax is None:
        ax[0].axhspan(bounds['states']['lower']['C_a'],bounds['states']['upper']['C_a'],
            facecolor=mplconfig.colors[2], alpha=0.3
        )
    ax[0].plot(t_pred, controller.data.prediction(('_x','C_a'))[0], ls='--', label='$C_{a,pred}$', color=line_ca[0].get_color(), alpha=.5)
    ax[0].set_ylabel('$c_A$ [mol/l]')
    ax[0].set_ylim(None, 2.5)

    # C_b
    if fig_ax is None:
        ax[1].axhspan(bounds['states']['lower']['C_b'],bounds['states']['upper']['C_b'],
            facecolor=mplconfig.colors[2], alpha=0.3
        )
    line_cb = ax[1].plot(t, simulator.data['_x','C_b'], label='$C_a$')
    ax[1].plot(t, controller.data['_tvp','C_b_set'], ls='-', label='$C_{b,set}$', color='k', alpha=.5)
    ax[1].plot(t_pred, controller.data.prediction(('_x','C_b'))[0], ls='--', label='$C_{b,pred}$', color=line_cb[0].get_color(), alpha=.5)
    ax[1].plot(t_pred, controller.data.prediction(('_tvp','C_b_set'))[0], ls='--', label='$C_{b,pred}$', color='k', alpha=.5)
    ax[1].set_ylabel('$c_B$ [mol/l]')
    ax[1].set_ylim(None, 2.5)

    # T_R
    line_tr = ax[2].plot(t, simulator.data['_x','T_R'], label='$T_R$')
    if fig_ax is None:
        ax[2].axhspan(bounds['states']['lower']['T_R'],bounds['states']['upper']['T_R'],
            facecolor=mplconfig.colors[2], alpha=0.3
        )
        ax[2].text(t[0], 112, 'feasible region', color=mplconfig.colors[2], alpha=1)
    ax[2].plot(t_pred, controller.data.prediction(('_x','T_R'))[0], ls='--', label='$T_R$', color=line_tr[0].get_color(), alpha=.5)
    ax[2].set_ylim(110, 140)
    ax[2].set_ylabel('$T_R$ [$^\circ$C]')

    # Seperate past and future
    for ax_i in ax:
        ax_i.axvline(t[-1], ls='--', color=mplconfig.colors[3], alpha=.5)

    ax[0].text(t[-1]-2*t[1], 0.3, r'$\leftarrow$ past', horizontalalignment='right')
    ax[0].text(t[-1]+2*t[1], 0.3, r'pred. $\rightarrow$', horizontalalignment='left')

    fig.align_ylabels()

    return fig, ax



# %% [markdown]
# ## Investigation I: Nominal MPC without uncertainties.

# %% 

mpc_1, tvp_1 = cstr_controller.get_mpc(model, bound_dict)
sim_1 = cstr_simulator.get_simulator(model)


tvp_1['_tvp', :, 'C_b_set'] = 1

run_closed_loop(mpc_1, sim_1, N_sim = 50)
# %% [markdown]
# ## Investigation II: Robust MPC without uncertainties.

# %%
mpc_2, tvp_2 = cstr_controller.get_mpc(model, bound_dict, overwrite_settings = {'n_robust': 1})
sim_2 = cstr_simulator.get_simulator(model)


tvp_2['_tvp', :, 'C_b_set'] = 1

run_closed_loop(mpc_2, sim_2, N_sim = 50)
# %% [markdown]
# ## Plot comparison

# %%
fig, ax = plot_results(mpc_2, sim_2, bound_dict)

plot_results(mpc_1, sim_1, bound_dict, fig_ax = (fig, ax))
line1 = ax[1].plot([],[], ls='-', label='Nominal', color=mplconfig.colors[1])
line2 = ax[1].plot([],[], ls='-', label='Robust', color=mplconfig.colors[0])
line3 = ax[1].plot([],[], ls='-', label='setpoint', color='k')
ax[-1].set_xlabel('Time [h]')
fig.tight_layout(pad=0)

ax[1].legend(line1+line2+line3, ['Nominal', 'Robust', 'setpoint'], fontsize=7, ncols=3, loc='upper left')


savepath = os.path.join(plot_path, 'results')
fig.savefig(os.path.join(savepath, '03_robust_vs_nominal_exact_parameters.pgf'))

# %% [markdown]

# ## Investigation III: Nominal MPC with uncertainties.

# %%

mpc_3, tvp_3 = cstr_controller.get_mpc(model, bound_dict, overwrite_settings = {'n_robust': 0})
sim_3 = cstr_simulator.get_simulator(model)

p_sim_3 = sim_3.p_fun(0)

p_sim_3['alpha'] = .96
p_sim_3['beta'] = 1.02

run_closed_loop(mpc_3, sim_3, N_sim = 50)
# %% [markdown]

# ## Investigation IV: Robust MPC with uncertainties.

# %%

mpc_4, tvp_4 = cstr_controller.get_mpc(model, bound_dict, overwrite_settings = {'n_robust': 1})
sim_4 = cstr_simulator.get_simulator(model)

p_sim_4 = sim_4.p_fun(0)

p_sim_4['alpha'] = .96
p_sim_4['beta'] = 1.02

run_closed_loop(mpc_4, sim_4, N_sim = 50)
# %%

fig, ax = plot_results(mpc_4, sim_4, bound_dict)

plot_results(mpc_3, sim_3, bound_dict, fig_ax = (fig, ax))
line1 = ax[1].plot([],[], ls='-', label='Nominal', color=mplconfig.colors[1])
line2 = ax[1].plot([],[], ls='-', label='Robust', color=mplconfig.colors[0])
line3 = ax[1].plot([],[], ls='-', label='setpoint', color='k')
ax[-1].set_xlabel('Time [h]')
fig.tight_layout(pad=0)

ax[1].legend(line1+line2+line3, ['Nominal', 'Robust', 'setpoint'], fontsize=7, ncols=3, loc='upper left')
fig.savefig(os.path.join(savepath, '03_robust_vs_nominal_uncertain_parameters.pgf'))
# %%
