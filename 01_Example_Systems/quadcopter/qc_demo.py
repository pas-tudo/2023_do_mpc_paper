# %% [markdown]
# # Demo file for the Quadcopter example
# %%

import qcmodel
import qccontrol
import plot_results
import qctrajectory
import importlib
import numpy as np
import matplotlib.pyplot as plt
from casadi import *

importlib.reload(qccontrol)
importlib.reload(qcmodel)
importlib.reload(plot_results)


plt.ion()

# %%

t_step = 0.02
qcconf = qcmodel.QuadcopterConfig()
simulator, sim_p_template = qccontrol.get_simulator(t_step, qcmodel.get_model(qcconf, with_pos=True))


# %%

mpc, mpc_p_template = qccontrol.get_MPC(t_step, qcmodel.get_model(qcconf, with_pos=True))
mpc.set_initial_guess()

# %%

figure_eight_trajectory = qctrajectory.get_wobbly_figure_eight(s=1.5, a=-1.5, height=1, wobble=0.2, rot=np.pi/2)

res_plot = plot_results.ResultPlot(qcconf, simulator.data, figsize=(12,8))
simulator.x0['pos'] = np.random.uniform(np.array([-2, -2, 0.]), np.array([2, 2, 1.5]))
simulator.x0['phi'] = np.random.uniform(-np.pi/8*np.ones(3), np.pi/8*np.ones(3))
simulator.reset_history()
mpc.reset_history()

qccontrol.mpc_fly_trajectory(
    simulator, 
    mpc, 
    mpc_p_template,
    sim_p_template, 
    N_iter=200, 
    callbacks=[res_plot.draw], 
    trajectory=figure_eight_trajectory,
    noise_dist=2e-3)
