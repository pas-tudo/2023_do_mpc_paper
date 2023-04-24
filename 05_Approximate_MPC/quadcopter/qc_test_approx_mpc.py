# %%
import importlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import sys
import os

sys.path.append(os.path.join('..','..','01_Example_Systems','quadcopter'))

import qcmodel
import qccontrol
import plot_results
import qctrajectory

importlib.reload(qccontrol)
importlib.reload(qcmodel)
importlib.reload(plot_results)

plt.ion()

# %%

t_step = 0.05
qcconf = qcmodel.QuadcopterConfig()
simulator, sim_p_template = qccontrol.get_simulator(t_step, qcmodel.get_model(qcconf, with_pos=True))


# %%

# Load keras model
keras_model = keras.models.load_model(os.path.join('models','qc_approx_mpc_model'))

keras_model.layers[-1].invert = True


class ApproxMPC:
    def __init__(self, keras_model):
        self.keras_model = keras_model

        self.n_x = 12
        self.n_u = 4

        self.x0 = np.zeros((self.n_x,1))
        self.u0 = np.zeros((self.n_u,1))

    def __call__(self, x0, p):

        x0[:3] = x0[:3]-p[:3]

        self.x0 = x0

        nn_in = np.concatenate((x0, self.u0, p[-1].reshape(-1,1)), axis=0).reshape(1,-1)

        u0 = np.clip(self.keras_model([nn_in]).numpy().reshape(-1,1), 0, 0.3)


        self.u0 = u0

        return u0


ampc = ApproxMPC(keras_model)


tracjectory = qctrajectory.get_wobbly_figure_eight(s=.5, a=1, height=.5, wobble=0., rot=0)

res_plot = plot_results.ResultPlot(qcconf, simulator.data, figsize=(12,8))
simulator.x0 = np.zeros((12,1))
simulator.reset_history()

x0 = simulator.x0.cat.full()

N_iter = 200

for k in range(N_iter):
    traj_setpoint = tracjectory(simulator.t0).T
    sim_p_template['pos_setpoint'] = traj_setpoint[:3] 
    sim_p_template['yaw_setpoint'] = traj_setpoint[-1]

    u0 = ampc(x0, traj_setpoint)
    x0 = simulator.make_step(u0)

    res_plot.draw()