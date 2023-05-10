# %% [markdown]
# # Quadcopter Control

# %%
# Essentials
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import pdb

# Tools
from IPython.display import clear_output
import copy
import sys
import importlib
import time

# Typing information
from typing import Tuple, List, Dict, Union, Optional, Any, Callable

# Specialized packages
from casadi import *
from casadi.tools import *

# Control packages
import do_mpc

# Quadcopter 
import qcmodel
from qctrajectory import Trajectory
    

# %% 
# Default global variables

_variance_state_noise =  np.array([
            1e-2, 1e-2, 1e-2,
            0,0,0,
            0,0,0,
            0,0,0
        ]).reshape(-1,1)

# %%
# Generate simulator for the quadcopter model
def get_simulator(t_step: float, model: do_mpc.model.Model) -> do_mpc.simulator.Simulator:
    """
    Create a simulator for the quadcopter model.
    """
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step = t_step)

    # Pass dummy tvp function (these wont affect the simulation)
    sim_p_template = simulator.get_p_template()
    simulator.set_p_fun(lambda t: sim_p_template)

    simulator.setup()

    return simulator, sim_p_template
# %%
def get_LQR(t_step: float, model: do_mpc.model.Model,
        Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None) -> do_mpc.controller.LQR:
    """
    Create a LQR estimator for the quadcopter model.
    """

    pos_setpoint = np.ones(3).reshape(-1,1)
    x_ss, u_ss = qcmodel.get_stable_point(model, p=1)

    linearmodel = do_mpc.model.linearize(model, x_ss, u_ss).discretize(t_step)

    # Discrete time inifinite horizon LQR
    lqr = do_mpc.controller.LQR(linearmodel)
    lqr.set_param(n_horizon = None, t_step=t_step)

    # Default Q and R matrices or values from argument
    if Q is None:
        Q = np.diag(np.ones(model.n_x))
    elif Q.shape != (model.n_x, model.n_x):
        raise ValueError("Q must be a {n_x}x{n_x} matrix.".format(nx=model.n_x))
    else:
      pass
    if R is None:
        R = 1e-2*np.eye(4)
    elif R.shape != (model.n_u, model.n_u):
        raise ValueError("R must be a {nu}x{nu} matrix.".format(nu=model.n_u))

    lqr.set_objective(Q = Q, R = R)
    lqr.setup()

    lqr.set_setpoint(x_ss, u_ss)

  
    return lqr

# %%

def get_MPC(t_step: float, model: do_mpc.model.Model) -> Tuple[do_mpc.controller.MPC, Any]:
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 30,
        't_step': t_step,   
        'store_full_solution': False,
        # Use MA27 linear solver in ipopt for faster calculations:
        'nlpsol_opts': {'ipopt.linear_solver': 'mumps', 'ipopt.tol': 1e-16}
    }
    mpc.set_param(**setup_mpc)

    surpress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
    mpc.set_param(nlpsol_opts = surpress_ipopt)

    p_template = mpc.get_p_template(n_combinations=1)
    def p_fun(t_now):
        return p_template
    mpc.set_p_fun(p_fun)


    lterm = 0 
    lterm += 10*sum1((model.x['pos'])**2)
    lterm += .01*sum1((model.x['dpos'])**2)
    lterm += .05*sum1((model.x['omega'])**2)
    lterm += 0.1*sum1((model.x['phi'])**2)
    # lterm += 1*(model.x['phi',0]**2)

    mterm = lterm
    mpc.set_objective(lterm=lterm, mterm=mterm)
    mpc.set_rterm(thrust=5)

    mpc.bounds['lower','_u','thrust'] = 0
    mpc.bounds['upper','_u','thrust'] = 0.3

    mpc.setup()

    mpc.x0 = np.ones((model.n_x,1))*1e-2
    mpc.set_initial_guess()

    return mpc, p_template


# %%
# %%

def lqr_flyto(
        simulator: do_mpc.simulator.Simulator, 
        lqr: do_mpc.controller.LQR, 
        v_x: Union[np.ndarray, float] = None,
        pos_setpoint: Optional[np.ndarray] = np.ones((3,1)),
        N_iter: Optional[int] = 60,
    ) -> Tuple[plt.Figure, plt.axes]:

    if v_x is None:
        v_x = _variance_state_noise

    
    # Update setpoint
    x_ss = np.zeros((12,1))
    x_ss[:3,:] = pos_setpoint

    lqr.set_setpoint(xss = x_ss, uss = lqr.uss)
    x0 = simulator.x0

    for k in range(N_iter):
        u0 = lqr.make_step(x0)
        u0 = np.clip(u0, 0, 0.3)
        x0 = simulator.make_step(u0) + np.random.randn(12,1)*v_x


def mpc_flyto(
        simulator: do_mpc.simulator.Simulator, 
        mpc: do_mpc.controller.MPC, 
        p_template: Any,
        v_x: Union[np.ndarray, float] = None,
        pos_setpoint: Optional[np.ndarray] = np.ones((3,1)),
        N_iter: Optional[int] = 60,
        ) -> None:

    p_template['_p',0] = 0 # Reset all setpoints
    p_template['_p',0, 'pos_setpoint'] = pos_setpoint 

    if v_x is None:
        v_x = _variance_state_noise

    x0 = simulator.x0
    for k in range(N_iter):
        u0 = mpc.make_step(x0)
        x0 = simulator.make_step(u0) + np.random.randn(12,1)*v_x
        

def mpc_fly_trajectory(
        simulator: do_mpc.simulator.Simulator,
        controller: Any,  
        sim_p_template: Any,
        trajectory: Trajectory,
        v_x: Union[np.ndarray, float] = None,
        N_iter: Optional[int] = 200,
        callbacks: Optional[List[Callable]] = [],
        noise_dist = 0.0,
        ) -> None:

    if v_x is None:
        v_x = _variance_state_noise

    x0 = simulator.x0.cat.full()
    for k in range(N_iter):
        traj_setpoint = trajectory(simulator.t0).T
        sim_p_template['pos_setpoint'] = traj_setpoint[:3] 
        sim_p_template['yaw_setpoint'] = traj_setpoint[-1]
        x0[:3] = x0[:3]-traj_setpoint[:3]

        u0 = controller.make_step(x0)

        u0 += np.random.uniform(-noise_dist, noise_dist, size=(4,1))

        u0 = np.clip(u0, 0, 0.3)

        x0 = simulator.make_step(u0)

        for cb in callbacks:
            cb()

        if np.any(simulator.x0.cat.full()>50):
            break

        time.sleep(.04)