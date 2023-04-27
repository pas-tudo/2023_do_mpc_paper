# %% [markdown]
# # Demo file for the Quadcopter example
# %%
import os
import sys

sys.path.append(os.path.join('..','..','01_Example_Systems','quadcopter'))

import qcmodel
import qccontrol
import plot_results
import qctrajectory
import importlib
import numpy as np
import matplotlib.pyplot as plt
import casadi.tools as ca
import do_mpc

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


mpc, mpc_p_template = qccontrol.get_MPC(t_step, qcmodel.get_model(qcconf, with_pos=True))
mpc.set_initial_guess()

nlp_diff = do_mpc.differentiator.DoMPCDifferentiatior(mpc)
nlp_diff.settings.lin_solver = 'casadi'
nlp_diff.settings.check_LICQ = False
nlp_diff.settings.check_rank = False
nlp_diff.settings.check_SC = False


# %%
t1 = do_mpc.tools.Timer()
t2 = do_mpc.tools.Timer()


trajectory = qctrajectory.get_wobbly_figure_eight(s=1.5, a=-1.5, height=1, wobble=0.2, rot=np.pi/2)


simulator.x0 = np.random.randn(12,1)
simulator.reset_history()
mpc.reset_history()

x0 = simulator.x0.cat.full()
for k in range(5):
        traj_setpoint = trajectory(mpc.t0).T
        sim_p_template['pos_setpoint'] = traj_setpoint[:3] 
        sim_p_template['yaw_setpoint'] = traj_setpoint[-1]
        mpc_p_template['_p', 0, 'yaw_setpoint'] = traj_setpoint[-1]
        x0[:3] = x0[:3]-traj_setpoint[:3]

        t1.tic()
        u0 = mpc.make_step(x0)
        t1.toc()
        x0 = simulator.make_step(u0)
        t2.tic()
        nlp_diff.differentiate()
        t2.toc()
# %%
t1.info()
t2.info()


# %%
nlp_diff.sens_num["dxdp",ca.indexf["_u",0,0], ca.indexf["_u_prev"]]
# %%
