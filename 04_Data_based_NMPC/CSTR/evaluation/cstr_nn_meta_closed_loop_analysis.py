# %% [markdown]
"""
# Analysis of closed-loop MPC results for the CSTR system with a neural network model
Comparison of controller with exact model and NN model. 

Import the necessary packages.
"""
# %% [code]
# Import packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import time as time
import sys 
import os
import json
import pathlib
import multiprocessing as mp
import importlib

import do_mpc
# %%

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
# %%
# df_res = pd.DataFrame(dh.filter(output_filter = lambda exact_mpc_success: exact_mpc_success > 0 ))
df_res = pd.DataFrame(dh[:])
df_res.sort_values(by='C_b_set', inplace=True)

# %%
# of type scatter
fig, ax = plt.subplots(2,1, sharex=True)

ax[0].plot(df_res['C_b_set'], df_res['exact_mpc_cost'], 'o', label='exact model', color='k', markerfacecolor='none')
ax[0].plot(df_res['C_b_set'], df_res['nn_mpc_cost'], 'x', label='NN model', color='k')
ax[0].legend(title='MPC with:', loc='lower right')

ax[0].set_ylabel('closed-loop cost [-]')

exact_mpc_cons_viol = np.concatenate(df_res['exact_mpc_cons_viol'].apply(lambda x: np.percentile(x,90, axis=0, keepdims=True)))
nn_mpc_cons_viol = np.concatenate(df_res['nn_mpc_cons_viol'].apply(lambda x: np.percentile(x,90, axis=0, keepdims=True)))

ax[1].semilogy(df_res['C_b_set'], exact_mpc_cons_viol, 'o',label=['c_a','c_b','T_R','T_K'])
ax[1].set_prop_cycle(None)
ax[1].semilogy(df_res['C_b_set'], nn_mpc_cons_viol, 'x')

ax[1].set_ylabel('90-percentile\n constraint violation')
ax[1].legend(ncols=4, loc='lower right')

fig.align_ylabels()
fig.tight_layout()



# %%

nn_mpc_cons_viol# %%

# %%
