# %%
import importlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import sys
import os
import time

sys.path.append(os.path.join('..','..','01_Example_Systems','quadcopter'))

import qcmodel
import qccontrol
import plot_results
import qctrajectory
from qc_approx_helper import ApproxMPC

importlib.reload(qccontrol)
importlib.reload(qcmodel)
importlib.reload(plot_results)

plt.ion()

# %%

t_step = 0.04
qcconf = qcmodel.QuadcopterConfig()
simulator, sim_p_template = qccontrol.get_simulator(t_step, qcmodel.get_model(qcconf, with_pos=True))


# %%

# Load keras model
keras_model = keras.models.load_model(os.path.join('meta_analysis','models_meta_02', 'qc_meta_traj_20_gamma_0'))
# keras_model.layers[-1].invert = True

ampc = ApproxMPC(keras_model, n_u=4, n_x=12, u0=0.067*np.ones((4,1)))


tracjectory = qctrajectory.get_wobbly_figure_eight(s=1, a=1, height=0, wobble=.5, rot=0)

res_plot = plot_results.ResultPlot(qcconf, simulator.data, figsize=(12,8))

# simulator.x0['pos'] = tracjectory(0).T[:3]
simulator.x0['phi'] = np.random.uniform(-np.pi/8*np.ones(3), np.pi/8*np.ones(3))
simulator.reset_history()

np.random.seed(42)

def update(k):
    qccontrol.mpc_fly_trajectory(
        simulator, 
        ampc, 
        sim_p_template, 
        N_iter=1, 
        callbacks=[res_plot.draw,], 
        trajectory=tracjectory,
        noise_dist=0.01)


from matplotlib.animation import FuncAnimation, ImageMagickFileWriter

fig = res_plot.fig
frames = np.arange(200)
anim = FuncAnimation(fig, update, frames=frames, interval=1)

writer = ImageMagickFileWriter(fps=25, bitrate=600)
anim.save('quadcopter_approx_mpc_wo_sobolev.gif', writer=writer)
