# %% [markdown]
# # Analysis of closed-loop MPC results for the CSTR system with a neural network model
# Comparison of controller with exact model and NN model. 
# Import the necessary packages.

# %% [code]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import time as time
import sys 
import os
import json
import importlib

import do_mpc

example_path = os.path.join('..','..','..','01_Example_Systems','CSTR')
plot_path = os.path.join('..','..','..','00_plotting')
sys.path.append(example_path)
sys.path.append(plot_path)
import cstr_helper
import mplconfig

importlib.reload(mplconfig)

mplconfig.config_mpl(os.path.join('..','..','..','00_plotting','notation.tex'))

# %% [markdown]
# # Load closed-loop data
# - Load the sampling plan
# - Initialize the data handler with the sampling plan
# - Load the data from the closed-loop samples

# %%
data_dir = os.path.join('.', 'closed_loop_meta')

plan = do_mpc.tools.load_pickle(os.path.join(data_dir, 'sampling_plan_closed_loop_meta.pkl'))

dh = do_mpc.sampling.DataHandler(plan)
dh.data_dir = os.path.join(data_dir, '')

# %% [markdown]
# # Post-preocessing of the results

bound_dict = json.load(open(os.path.join('..','..','..','01_Example_Systems','CSTR', 'config','cstr_bounds.json')))

def cost(C_b_set, C_b):
    return np.sum((C_b_set - C_b)**2)

def cons_viol(res, bound_dict):

    x_lb = np.array([val for val in bound_dict['states']['lower'].values()])
    x_ub = np.array([val for val in bound_dict['states']['upper'].values()])

    lb_viol = np.maximum(x_lb - res['_x'], 0)
    ub_viol = np.maximum(res['_x'] - x_ub, 0)

    cons_viol = lb_viol + ub_viol

    return cons_viol

def perc_success(res):
    return np.sum(res['success'])/len(res['success'])



dh.set_post_processing('exact_mpc_cost',
    lambda res: cost(res['mpc_true']['_x', 'C_b'], res['mpc_true']['_tvp', 'C_b_set'])
)

dh.set_post_processing('nn_mpc_cost',
    lambda res: cost(res['mpc_nn']['_x', 'C_b'], res['mpc_nn']['_tvp', 'C_b_set'])
)

dh.set_post_processing('exact_mpc_cons_viol',
    lambda res: cons_viol(res['mpc_true'], bound_dict)
)

dh.set_post_processing('nn_mpc_cons_viol',
    lambda res: cons_viol(res['mpc_nn'], bound_dict)
)

dh.set_post_processing('exact_mpc_success',
    lambda res: perc_success(res['mpc_true'])
)

# %% [markdown]
# ## Convert to pandas DataFrame


# %%
df_res = pd.DataFrame(dh.filter(output_filter = lambda exact_mpc_success: exact_mpc_success > 0.99 ))
df_res.sort_values(by='C_b_set', inplace=True)

# %% [markdown]
# ## Plot results

# %%
fig, ax = plt.subplots(2,1, figsize=(mplconfig.columnwidth, 1.1*mplconfig.columnwidth), dpi=160, sharex=True)

ax[0].semilogy(df_res['C_b_set'], df_res['exact_mpc_cost'], 'o', label='exact model', color='k', markerfacecolor='none')
ax[0].semilogy(df_res['C_b_set'], df_res['nn_mpc_cost'], 'x', label='NN model', color='k')
ax[0].legend(title='MPC with:', loc='upper left', fontsize='small')

ax[0].set_ylabel('closed-loop cost [-]')

exact_mpc_cons_viol = np.concatenate(df_res['exact_mpc_cons_viol'].apply(lambda x: np.percentile(x,100, axis=0, keepdims=True)))
nn_mpc_cons_viol = np.concatenate(df_res['nn_mpc_cons_viol'].apply(lambda x: np.percentile(x,100, axis=0, keepdims=True)))

ax[1].semilogy(df_res['C_b_set'], exact_mpc_cons_viol, 'o',label=['$c_A$ [-]','$c_B$ [-]','$T_R$ [°C]','$T_K$ [°C]'], markerfacecolor='none')
ax[1].set_prop_cycle(None)
ax[1].semilogy(df_res['C_b_set'], nn_mpc_cons_viol, 'x')

ax[1].set_ylabel('max. constraint violation')
ax[1].set_xlabel(r'$c_B^\text{set}$ [-]')
ax[1].legend(loc='lower left', fontsize='small', ncols=2)

fig.align_ylabels()
fig.tight_layout(pad=0.1)

fig.savefig(os.path.join(plot_path, 'results', 'mpc_with_nn_vs_exact_model_meta.pgf'))

# %%
