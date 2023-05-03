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
import logging
import matplotlib.pyplot as plt
import casadi.tools as ca
import do_mpc

importlib.reload(qccontrol)
importlib.reload(qcmodel)
importlib.reload(plot_results)

logging.basicConfig( level=logging.INFO)

plt.ion()

# %%

t_step = 0.04
qcconf = qcmodel.QuadcopterConfig()
simulator, sim_p_template = qccontrol.get_simulator(t_step, qcmodel.get_model(qcconf, with_pos=True))


mpc, mpc_p_template = qccontrol.get_MPC(t_step, qcmodel.get_model(qcconf, with_pos=True))
mpc.set_initial_guess()


# %%
t1 = do_mpc.tools.Timer()
t2 = do_mpc.tools.Timer()


trajectory = qctrajectory.get_wobbly_figure_eight(s=1.5, a=-1.5, height=1, wobble=0.2, rot=np.pi/2)


simulator.x0['pos'] = np.random.uniform(np.array([-1, -1, 0.5]), np.array([1, 1, 1.5]))
simulator.x0['phi'] = np.random.uniform(-np.pi/8*np.ones(3), np.pi/8*np.ones(3))
simulator.reset_history()
mpc.reset_history()

class ASMPC:
    def __init__(self, mpc):
        self.mpc = mpc
        self.nlp_diff = do_mpc.differentiator.DoMPCDifferentiatior(mpc)
        self.nlp_diff.settings.check_LICQ = True
        self.nlp_diff.settings.check_rank = False
        self.nlp_diff.settings.check_SC = True
        self.nlp_diff.settings.solver ='casadi'

        self._u_data = [mpc.u0.cat.full().reshape(-1,1)]
        self._licq = []

    def make_step(self, x0):
        x0 = x0.reshape(-1,1)

        self.nlp_diff.differentiate()

        self._licq.append(self.nlp_diff.status.LICQ)


        x_prev = self.mpc.x0.cat.full().reshape(-1,1)
        u0 = self.mpc.u0.cat.full().reshape(-1,1)
        u_prev = self.mpc.opt_p_num['_u_prev'].full().reshape(-1,1)

        

        du0dx0_num = self.nlp_diff.sens_num["dxdp", ca.indexf["_u",0,0], ca.indexf["_x0"]]
        du0du_prev_num = self.nlp_diff.sens_num["dxdp", ca.indexf["_u",0,0], ca.indexf["_u_prev"]].full()

        A = np.eye(self.mpc.model.n_u)-du0du_prev_num

        u_next  = np.linalg.inv(A)@(u0 + du0dx0_num @ (x0 - x_prev) - du0du_prev_num @ (u0))
        # u_next = u0 + du0dx0_num @ (x0 - x_prev) - du0du_prev_num @ (u0 - u_prev)

        self._u_data.append(u_next)

        return u_next
    
    @property
    def u_data(self):
        return np.hstack(self._u_data)
    
    @property
    def licq(self):
         return np.array(self._licq)




asmpc = ASMPC(mpc)


x0 = simulator.x0.cat.full()
for k in range(50):
        print(k, end='\r')
        traj_setpoint = trajectory(mpc.t0).T
        sim_p_template['pos_setpoint'] = traj_setpoint[:3] 
        sim_p_template['yaw_setpoint'] = traj_setpoint[-1]
        mpc_p_template['_p', 0, 'yaw_setpoint'] = traj_setpoint[-1]
        x0[:3] = x0[:3]-traj_setpoint[:3]

        if k > 0:
            asmpc.make_step(x0)
        t1.tic()
        u0 = mpc.make_step(x0)
        t1.toc()

        u0 += np.random.uniform(np.zeros(4), np.ones(4)*2e-3).reshape(-1,1)
        u0 = np.clip(u0, np.zeros((4,1)), 0.3*np.ones((4,1)))

        x0 = simulator.make_step(u0)
# %%
t1.info()


# %%
fig, ax = plt.subplots(5,1)

for k in range(4):
    ax[k].plot(asmpc.u_data[k], '-x', label="approx")
    ax[k].plot(mpc.data['_u'][:,k], '-x', label="mpc")

ax[0].plot(asmpc.licq, '-o', label='licq')
ax[0].plot(mpc.data['success'], '-x', label='success')
ax[0].legend()

ax[-1].plot(simulator.data['_x', 'pos'])
ax[-1].set_prop_cycle(None)
ax[-1].plot(simulator.data['_p', 'pos_setpoint'], '--')

plt.show(block=True)
# %%
