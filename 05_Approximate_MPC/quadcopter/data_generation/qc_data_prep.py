# %% [markdown]
# # Create data set for approximate MPC
# 
# Load data from the generated closed-loop samples and prepare it for the approximate MPC. 

# %%
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
from functools import partial
import pdb
import time

# Control packages
import do_mpc

# %% [markdown]
# ## Create helper functions for data handling and visualization

# %%
def analyze_success_and_licq(results):
    print('Optimizer success in for all samples of case:')
    for i, res in enumerate(results):
        print('Case {}: {}'.format(i,np.all(res['res']['success'])))

    print('LICQ satisfied in for all samples of case (None means not tested):')
    for i, res in enumerate(results):
        print('Case {}: {}'.format(i,np.all(res['res']['nlp_licq'])))

 # %%
def animate_results(results):
    plt.ion()
    fig, ax = plt.subplots()

    lines = []

    for res in results:
        for i, line in enumerate(lines):
            line[0].set_alpha(1/(i+1))

        #lines.append(ax.plot(res['res']['x_k'][:,0], res['res']['x_k'][:,1], color='k'))
        lines.append(ax.plot(res['res']['pos_k'][:,0], res['res']['pos_k'][:,1], '--', color='r', ))
        plt.show()
        plt.pause(0.5)

    return fig, ax

# %%

def plot_some_trajectories(results, n_traj=10):
    fig, ax = plt.subplots(n_traj, sharex=True)

    for i, res in enumerate(results[:n_traj]):
        ax[i].plot(res['res']['pos_k'][:,0], label='x')
        ax[i].plot(res['res']['pos_k'][:,1], label='y')
        ax[i].plot(res['res']['pos_k'][:,2], label='z')

        ax[i].set_prop_cycle(None)

        ax[i].plot(res['res']['p_k'][:,0], '--', label='x set')
        ax[i].plot(res['res']['p_k'][:,1], '--', label='y set')
        ax[i].plot(res['res']['p_k'][:,2], '--', label='z set')

        ax[i].set_ylabel('pos.')

    ax[-1].legend()

    return fig, ax

# %%
    
def get_data_for_approx_mpc(results):
    X_K = []
    U_K = []
    U_K_prev = []
    dUdX0 = []
    dUdU0_prev = []


    for res_i in results:
        # dyaw =np.sin(res_i['res']['p_k'][:,[-1]] - res_i['res']['x_k'][:, [6]])
        X_K.append(res_i['res']['x_k'][1:,:])
        U_K.append(res_i['res']['u_k'][1:,:])
        U_K_prev.append(res_i['res']['u_k'][:-1,:])
        # P_k.append(res_i['res']['p_k'][1:,[-1]])
        # P_k.append(dyaw[1:,:])
        dUdX0.append(res_i['res']['du0dx0'][1:])
        dUdU0_prev.append(res_i['res']['du0du0_prev'][1:])

    X_K = np.concatenate(X_K, axis=0)
    U_K = np.concatenate(U_K, axis=0)
    U_K_prev = np.concatenate(U_K_prev, axis=0)
    dUdX0 = np.concatenate(dUdX0, axis=0)
    dUdU0_prev = np.concatenate(dUdU0_prev, axis=0)
    # P_K = np.concatenate(P_k, axis=0)

    dUdX0 = [v for v in dUdX0]
    dUdU0_prev = [v for v in dUdU0_prev]

    data = pd.concat(
        [
            pd.DataFrame(X_K, columns=['dx0', 'dx1', 'dx2', 'v0', 'v1', 'v2', 'phi0', 'phi1', 'phi2', 'omega0', 'omega1', 'omega2']),
            pd.DataFrame(U_K, columns=['f0', 'f1', 'f2', 'f3']),
            pd.DataFrame(U_K_prev, columns=['f0', 'f1', 'f2', 'f3']),
            pd.DataFrame({'du0dx0': dUdX0}),
            pd.DataFrame({'dUdU0_prev': dUdU0_prev}),
        ],
        axis=1,
        keys=['x_k', 'u_k', 'u_k_prev', 'du0dx0', 'du0du0_prev']
    )

    return data


# %% [markdown]
# # Load closed-loop data
# - Load the sampling plan
# - Initialize the data handler with the sampling plan
# - Load the data from the closed-loop samples
# - Filter out the samples that did not converge
# - Visualize the results

# %%
if __name__ == '__main__':

    data_dir = os.path.join('.', 'closed_loop_mpc')

    plan = do_mpc.tools.load_pickle(os.path.join(data_dir, 'sampling_plan_mpc.pkl'))

    dh = do_mpc.sampling.DataHandler(plan)
    dh.data_dir = os.path.join(data_dir, '')

    # Filter out the samples that did not converge
    results_with_success = dh.filter(output_filter = lambda res: res is not None)
    analyze_success_and_licq(results_with_success)

    # %% [markdown]
    # ## Visulations to analyze the data
    if False:
        animate_results(results_with_success)

    if True:
        plot_some_trajectories(results_with_success, 3)

    # %% [markdown]
    # ## Pre-process data 

    df_data = get_data_for_approx_mpc(results_with_success)


    # %% [markdown]
    # ## Analsis of the data
    # ### Description:

    # %% [code]

    df_data.describe()

    # %% [markdown]
    # ### Histograms

    # %%

    ax = df_data[['x_k', 'u_k']].hist(figsize=(8,8))
    fig = plt.gcf()
    fig.tight_layout()


    # %% [markdown]
    # ## Export df_data to pickle

    # %%
    df_data.to_pickle('qc_data_mpc.pkl')

    # %%
