
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

sys.path.append(os.path.join('..', '.'))
sys.path.append(os.path.join('..','..','..','01_Example_Systems','quadcopter'))

import qcmodel
import qccontrol
import plot_results
import qctrajectory
from qc_approx_helper import ApproxMPC


plot_path = os.path.join('..', '..','..','00_plotting')
sys.path.append(plot_path)
import mplconfig
mplconfig.config_mpl(os.path.join(plot_path,'notation.tex'))

plt.ion()
# %%
# Helper function
def get_config_from_name(model_name):
    model_name = model_name.split('_')
    conf = {}

    conf['test_perc'] = float(model_name[4])*10
    conf['gamma'] = float(model_name[-1])

    return conf


class ProgressCallback:
    def __init__(self):
        self.k = 1

    def __call__(self):
        print(f'Iteration: {self.k}', end='\r')
        self.k += 1

# Get simulator 
t_step = 0.04
qcconf = qcmodel.QuadcopterConfig()
simulator, sim_p_template = qccontrol.get_simulator(t_step, qcmodel.get_model(qcconf, with_pos=True))


# Prepare result dictionary 
res = []

# List dir in models_meta
model_name_list = os.listdir('models_meta')
noise_dist_list = [0, 1e-3, 1e-2]


for model_name in model_name_list:

    keras_model = keras.models.load_model(os.path.join('models_meta', model_name))
    simulator, sim_p_template = qccontrol.get_simulator(t_step, qcmodel.get_model(qcconf, with_pos=True))

    for noise_dist in noise_dist_list:
        print('-------------------')
        print(model_name)
        print(f'Noise dist: {noise_dist}')

        ampc = ApproxMPC(keras_model, n_u=4, n_x=12, u0=0.067*np.ones((4,1)))


        tracjectory = qctrajectory.get_wobbly_figure_eight(s=1, a=1, height=0, wobble=.5, rot=0)

        # res_plot = plot_results.ResultPlot(qcconf, simulator.data, figsize=(12,8))

        simulator.x0['pos'] = tracjectory(0).T[:3]
        simulator.x0['phi'] = np.random.uniform(-np.pi/8*np.ones(3), np.pi/8*np.ones(3))
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
            **get_config_from_name(model_name),
            'sim_res': copy.copy(simulator.data),
            'noise_dist': noise_dist,
        })
# %%
res_df = pd.DataFrame(res)
# %%
res_df['closed_loop_cost'] = res_df['sim_res'].apply(lambda x: t_step*np.sum(x['_aux', 'del_setpoint']))
res_df['n_iter'] = res_df['sim_res'].apply(lambda x: x['_aux', 'del_setpoint'].shape[0])
# %%

# %%

# fig, ax = plt.subplots(4,3, sharex=True, sharey=True, dpi=200)

fig_outer = plt.figure(layout='constrained', figsize=(mplconfig.columnwidth, 2.3), dpi=200)

ax_outer = fig_outer.add_axes([0,0,1,1])
ax_outer.set(frame_on=False)
fig_inner = fig_outer.subfigures(1,1, wspace=0., hspace=0.)

ax_outer.set_ylabel(r'increasing number of training samples $\longrightarrow$')
ax_outer.set_xlabel(r'increasing additive input disturbance $\longrightarrow$')

ax_outer.set_xlim(0,1)
ax_outer.set_xticks([], [])
ax_outer.set_yticks([], [])

ax = fig_inner.subplots(4,3, sharex=True, sharey=True)


for i, perc in enumerate(res_df['test_perc'].unique()):
    for j, noise_dist in enumerate(res_df['noise_dist'].unique()):
        filtered = res_df.loc[res_df['test_perc'] == perc].loc[res_df['noise_dist'] == noise_dist]

        with_sobo = filtered.loc[filtered['gamma'] >0 ].iloc[0]
        without_sobo = filtered.loc[filtered['gamma'] == 0 ].iloc[0]

        ax[i,j].plot(with_sobo['sim_res']['_x','pos'][:,0], with_sobo['sim_res']['_x','pos'][:,1], label='with sobolov')
        ax[i,j].plot(without_sobo['sim_res']['_x','pos'][:,0], without_sobo['sim_res']['_x','pos'][:,1], label='w/o sobolov')

        if with_sobo['n_iter'] > 100:
            val = with_sobo['closed_loop_cost']
            p = ax[i,j].bar(1.2,  val/2.5, bottom=-1,label=val, width=0.1)
            ax[i,j].bar_label(p, fmt='%.2f')
        if without_sobo['n_iter'] > 100:
            val = without_sobo['closed_loop_cost']
            p = ax[i,j].bar(1.4,  val/2.5,bottom=-1, label=val, width=0.1)
            ax[i,j].bar_label(p, fmt='%.2f')



        ax[i,j].axis('off')

    ax[-1,-1].legend()


# fig_inner.tight_layout(pad=0)

plt.show()


fig_outer.savefig(os.path.join(plot_path, 'results', '05_approx_mpc_sobolov.pgf'), bbox_inches='tight')


# %%


# %%