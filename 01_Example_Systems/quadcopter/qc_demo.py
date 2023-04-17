# %% [markdown]
# # Demo file for the Quadcopter example
# %%

import qcmodel
import qccontrol
from plot_results import plot_results
import importlib
import numpy as np
import matplotlib.pyplot as plt
from casadi import *

importlib.reload(qccontrol)
importlib.reload(qcmodel)
# %%

t_step = 0.05
qcconf = qcmodel.QuadcopterConfig()


simulator = qccontrol.get_simulator(t_step, qcmodel.get_model(qcconf, with_pos=True))

x0 = np.zeros((12,1))
simulator.x0 = x0
# %%

lqr = qccontrol.get_LQR(t_step, qcmodel.get_model(qcconf, with_pos=True))

# %%
qccontrol.lqr_flyto(simulator, lqr, pos_setpoint=np.array([1,2,1]).reshape(-1,1))
qccontrol.lqr_flyto(simulator, lqr, pos_setpoint=np.array([1,1,2]).reshape(-1,1))
qccontrol.lqr_flyto(simulator, lqr, pos_setpoint=np.array([-1,-1,1]).reshape(-1,1))
fig, ax = plot_results(qcconf, lqr.data, figsize=(12,8)) 

plt.show(block=True)
# %%
mpc, mpc_tvp_template = qccontrol.get_MPC(t_step, qcmodel.get_model(qcconf, with_pos=True))

# %%
simulator.reset_history()
simulator.x0 = np.zeros((12,1))


mpc.reset_history()
for k in range(50):
    mpc_tvp_template['_tvp', :, 'pos_setpoint'] = np.array([1, 1, 1]).reshape(-1,1)
    mpc.make_step(simulator.x0)

    simulator.make_step(mpc.u0)

# %%
fig, ax = plot_results(qcconf, simulator.data, figsize=(12,8)) 
# %%