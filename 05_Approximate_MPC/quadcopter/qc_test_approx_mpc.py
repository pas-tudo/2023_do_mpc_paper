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
keras_model = keras.models.load_model(os.path.join('models','03_qc_approx_mpc_model'))

keras_model.layers[-1].invert = True


class ApproxMPC:
    def __init__(self, keras_model):
        self.keras_model = keras_model

        self.n_x = 12
        self.n_u = 4

        self.x0 = np.zeros((self.n_x,1))
        self.u0 = np.ones((self.n_u,1))*.067

    def __call__(self, x0, p):

        dp_max = np.array([.3,.3,.1]).reshape(-1,1)

        x0[:3] = np.clip(x0[:3]-p[:3], -dp_max, dp_max)

        u_prev = self.u0


        self.x0 = x0



        nn_in = (
            x0.reshape(1,-1),
            u_prev.reshape(1,-1),
        )

        u0 = self.keras_model(nn_in).numpy().reshape(-1,1)


        self.u0 = u0

        return u0


ampc = ApproxMPC(keras_model)


tracjectory = qctrajectory.get_wobbly_figure_eight(s=1, a=1, height=1, wobble=1.5, rot=np.pi/2)

res_plot = plot_results.ResultPlot(qcconf, simulator.data, figsize=(12,8))
simulator.x0 = np.zeros((12,1))
simulator.reset_history()

x0 = simulator.x0.cat.full()

N_iter = 400

for k in range(N_iter):
    traj_setpoint = tracjectory(simulator.t0).T
    sim_p_template['pos_setpoint'] = traj_setpoint[:3] 
    # sim_p_template['yaw_setpoint'] = traj_setpoint[-1]

    u0 = ampc(x0, traj_setpoint)
    u0 += np.random.uniform(-2e-3*np.ones(4), 2e-3*np.ones(4)).reshape(-1,1)
    x0 = simulator.make_step(u0)

    if k % 10 == 0:
        res_plot.draw()

    if np.any(x0 > 100):
        break

    time.sleep(0.01)

plt.show(block=True)