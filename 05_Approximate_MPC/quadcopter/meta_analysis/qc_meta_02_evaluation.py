# %% [markdown]
# # Meta analysis of approximate MPC for quadcopter
# In this script, the trained approximate MPC controllers obtained in ``qc_meta_01.py`` are evaluated.
# The controller variants differ in the number of trajectories used for training and value of $\gamma$ for the Sobolev norm. 
# 
# First, we import the required packages.

# %%
import sys 
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from casadi import *
import copy
import pandas as pd
import json

# Add path for quadcopter files and import them
sys.path.append(os.path.join('..', '.'))
sys.path.append(os.path.join('..','..','..','01_Example_Systems','quadcopter'))
import qcmodel
import qccontrol
import plot_results
import qctrajectory
from qc_approx_helper import ApproxMPC

# Configure plotting
plot_path = os.path.join('..', '..','..','00_plotting')
sys.path.append(plot_path)
import mplconfig
mplconfig.config_mpl(os.path.join(plot_path,'notation.tex'))

plt.ion()

# %% [markdown]
# ## Preparation
# - Prepare setup of the simulator for the quadcopter model, representing the true system.
# - Get the list of models to be evaluated.
# %%
class ProgressCallback:
    def __init__(self):
        self.k = 1

    def __call__(self):
        print(f'Iteration: {self.k}', end='\r')
        self.k += 1

# Get simulator 
t_step = 0.04
qcconf = qcmodel.QuadcopterConfig()

# List dir in models_meta

model_path = 'models_meta_02'
model_name_list = os.listdir(model_path)
noise_dist_list = [0, 1e-3, 1e-2]

# %% [markdown]
# ## Evaluation
# The purpose of the meta evaluation is to test the performance of the trained approximate MPC controllers on the true system.
# for models trained on different number of trajectories and different values of $\gamma$. 
# Furthermore, the robustness of the controllers is tested by adding input noise to the quadcopter system.
# 
# The following steps are performed for each model:
# - Load the keras model and the meta data (containing the values of $\gamma$ and the number of trajectories used for training).
# - Generate a simulator
# - For each of the investigated noise magnitudes:
#     - Setup the approximate MPC controller
#     - Run the controller on the true system
#     - Store the results in a dictionary

# %%
# Fix the seed
np.random.seed(99)

# For all tests, follow this trajectory
tracjectory = qctrajectory.get_wobbly_figure_eight(s=1, a=1.0, height=0, wobble=.5, rot=0)

# Prepare result dictionary 
res = []

for model_name in model_name_list:

    keras_model = keras.models.load_model(os.path.join(model_path, model_name))
    with open(os.path.join(model_path, model_name, 'custom_meta.json')) as f:
        meta = json.load(f)

    simulator, sim_p_template = qccontrol.get_simulator(t_step, qcmodel.get_model(qcconf, with_pos=True))

    for noise_dist in noise_dist_list:
        print('-------------------')
        print(model_name)
        print(f'Noise dist: {noise_dist}')

        ampc = ApproxMPC(keras_model, n_u=4, n_x=12, u0=0.067*np.ones((4,1)))


        simulator.x0['pos'] = tracjectory(0).T[:3]
        simulator.reset_history()

        progress_cb = ProgressCallback()

        qccontrol.mpc_fly_trajectory(
            simulator, 
            ampc, 
            sim_p_template, 
            N_iter=200, 
            callbacks=[progress_cb,], 
            trajectory=tracjectory,
            noise_dist=noise_dist)

        res.append({
            'model_name': model_name,
            **meta,
            'sim_res': copy.copy(simulator.data),
            'noise_dist': noise_dist,
        })
# %% [markdown]
# ## Postprocessing
# The results are stored in a pandas dataframe and additional columns are added for further analysis.


# %%
res_df = pd.DataFrame(res)

# %%
res_df['closed_loop_cost'] = res_df['sim_res'].apply(lambda x: t_step*np.sum(x['_aux', 'del_setpoint']))
res_df['n_iter'] = res_df['sim_res'].apply(lambda x: x['_aux', 'del_setpoint'].shape[0])

# %%
res_df

# %% [markdown]
# ## Plotting
# Generate plots for each model and noise magnitude.

# %%

fig_outer = plt.figure(layout='constrained', figsize=(mplconfig.columnwidth, 3), dpi=200)

ax_outer = fig_outer.add_axes([0,0,1,1])
ax_outer.set(frame_on=False)
fig_inner = fig_outer.subfigures(1,1, wspace=0., hspace=0.)

ax_outer.set_ylabel(r'number of training trajectories $p$', fontsize=12)
ax_outer.set_xlabel(r'additive input disturbance $\alpha$ in test', fontsize=12)

ax_outer.set_xlim(0,1)
ax_outer.set_xticks([], [])
ax_outer.set_yticks([], [])


n_traj_list = np.flip(np.sort(res_df['number_of_trajectories'].unique()))
noise_dist_list = np.sort(res_df['noise_dist'].unique())

ax = fig_inner.subplots(len(n_traj_list), len(noise_dist_list), sharex=True, sharey=True)

for i, n_tra in enumerate(n_traj_list):
    for j, noise_dist in enumerate(noise_dist_list):
        filtered = res_df.loc[res_df['number_of_trajectories'] == n_tra].loc[res_df['noise_dist'] == noise_dist]

        with_sobo = filtered.loc[filtered['gamma_sobolov'] >0 ].iloc[0]
        without_sobo = filtered.loc[filtered['gamma_sobolov'] == 0 ].iloc[0]

        line_sobo = ax[i,j].plot(with_sobo['sim_res']['_x','pos'][:,0], with_sobo['sim_res']['_x','pos'][:,1], label='with Sobolov')
        line_mse = ax[i,j].plot(without_sobo['sim_res']['_x','pos'][:,0], without_sobo['sim_res']['_x','pos'][:,1], label='w/o sobolov')

        if False:
            if with_sobo['n_iter'] > 150:
                val = with_sobo['closed_loop_cost']
                p = ax[i,j].bar(1.2,  val/2.5, bottom=-1,label=val, width=0.1)
                ax[i,j].bar_label(p, fmt='%.2f')
            if without_sobo['n_iter'] > 150:
                val = without_sobo['closed_loop_cost']
                p = ax[i,j].bar(1.4,  val/2.5,bottom=-1, label=val, width=0.1)
                ax[i,j].bar_label(p, fmt='%.2f')

        ax[i,j].xaxis.set_ticks([])
        ax[i,j].yaxis.set_ticks([])
        ax[i,j].set_xlim(-1.1,1.1)
        ax[i,j].set_ylim(-1.1,1.1)
        ax[i,j].spines[['top', 'right', 'bottom', 'left']].set_visible(False)

        if i == 3:
            ax[i,j].set_xlabel(r'$\alpha$='+f'{noise_dist}', fontsize=10)

        # ax[i,j].axis('off')

    ax[i, 0].set_ylabel(f'$p=${n_tra}', fontsize=10)

    ax[-1,-1].legend(line_sobo+line_mse, [r'$L_\text{Sob}$', r'$L_\text{MSE}$'], loc='upper left')


plt.show()


fig_outer.savefig(os.path.join(plot_path, 'results', '05_approx_mpc_sobolov.pgf'), bbox_inches='tight')


# %% [markdown]
# ## Export results to table

# %%

idx = ['train_mse', 'val_mse', 'number_of_trajectories']

sobo_df = res_df[res_df['gamma_sobolov'] == 100][res_df['noise_dist']==0][idx].sort_values('number_of_trajectories')
mse_df = res_df[res_df['gamma_sobolov'] == 0][res_df['noise_dist']==0][idx].sort_values('number_of_trajectories')

sobo_df.index = sobo_df['number_of_trajectories']
sobo_df.drop('number_of_trajectories', axis=1, inplace=True)

mse_df.index = mse_df['number_of_trajectories']
mse_df.drop('number_of_trajectories', axis=1, inplace=True)

combined_df = pd.concat([sobo_df, mse_df], axis=1, keys=['sobolov', 'mse'])
combined_df *= 1e4

combined_df
# %%

combined_df.to_latex(
    os.path.join(plot_path, 'results', '05_approx_mpc_sobolov_table.tex'),
    float_format="%.2f",
)
# %%
