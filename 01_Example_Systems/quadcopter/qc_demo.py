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

demonstrate_lqr = False
demonstrate_mpc = True

plt.ion()

# %%

t_step = 0.05
qcconf = qcmodel.QuadcopterConfig()
simulator, sim_p_template = qccontrol.get_simulator(t_step, qcmodel.get_model(qcconf, with_pos=True))


# %%

lqr = qccontrol.get_LQR(t_step, qcmodel.get_model(qcconf, with_pos=True))

# %%
if demonstrate_lqr:
    x0 = np.zeros((12,1))
    simulator.x0 = x0
    qccontrol.lqr_flyto(simulator, lqr, pos_setpoint=np.array([1,2,1]).reshape(-1,1))
    qccontrol.lqr_flyto(simulator, lqr, pos_setpoint=np.array([1,1,2]).reshape(-1,1))
    qccontrol.lqr_flyto(simulator, lqr, pos_setpoint=np.array([-1,-1,1]).reshape(-1,1))
    fig, ax = plot_results(qcconf, simulator.data, figsize=(12,8)) 

# %%

mpc, mpc_p_template = qccontrol.get_MPC(t_step, qcmodel.get_model(qcconf, with_pos=True))
mpc.set_initial_guess()

# %%

figure_eight_trajectory = qctrajectory.get_wobbly_figure_eight(s=1.5, a=-1.5, height=1, wobble=0.2, rot=np.pi/2)

if demonstrate_mpc:
    res_plot = plot_results.ResultPlot(qcconf, simulator.data, figsize=(12,8))
    simulator.x0 = np.random.randn(12,1)
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
