# %% [markdown]
# # Quadcopter Control

# %%
# Essentials
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys

# Tools
from IPython.display import clear_output
import copy
import sys
import importlib

# Typing information
from typing import Tuple, List, Dict, Union, Optional, Any

# Specialized packages
from casadi import *
from casadi.tools import *

# Control packages
sys.path.append(os.path.join('..', '..', '..', 'do-mpc'))
import do_mpc

# Quadcopter 
import qcmodel
from plot_results import plot_results
    

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
def get_simulator(t_step: float, qc: qcmodel.Quadcopter) -> do_mpc.simulator.Simulator:
    """
    Create a simulator for the quadcopter model.
    """
    simulator = do_mpc.simulator.Simulator(qc.model)
    simulator.set_param(t_step = t_step)

    # Pass dummy tvp function (these wont affect the simulation)
    sim_tvp_template = simulator.get_tvp_template()
    simulator.set_tvp_fun(lambda t: sim_tvp_template)

    simulator.setup()

    return simulator
# %%
def get_LQR(t_step: float, qc: qcmodel.Quadcopter, 
        Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None) -> do_mpc.controller.LQR:
    """
    Create a LQR estimator for the quadcopter model.
    """

    pos_setpoint = np.ones(3).reshape(-1,1)
    x_ss, u_ss = qc.stable_point(pos_setpoint = pos_setpoint, p=1)

    linearmodel = do_mpc.model.linearize(qc.model, x_ss, u_ss).discretize(t_step)

    # Discrete time inifinite horizon LQR
    lqr = do_mpc.controller.LQR(linearmodel)
    lqr.set_param(n_horizon = None)

    # Default Q and R matrices or values from argument
    if Q is None:
        Q = np.diag(np.array([
            10,10,10, # Position
            1,1,1, # Velocity
            1,1,1, # Angle
            1,1,1, # Angular velocity
        ]))
    elif Q.shape != (qc.model.n_x, qc.model.n_x):
        raise ValueError("Q must be a {n_x}x{n_x} matrix.".format(nx=qc.model.n_x))
    else:
      pass
    if R is None:
        R = 1e-2*np.eye(4)
    elif R.shape != (qc.model.n_u, qc.model.n_u):
        raise ValueError("R must be a {nu}x{nu} matrix.".format(nu=qc.model.n_u))

    lqr.set_objective(Q = Q, R = R)
    lqr.setup()

    lqr.set_setpoint(x_ss, u_ss)

  
    return lqr

# %%

def get_MPC(t_step: float, qc: qcmodel.Quadcopter) -> Tuple[do_mpc.controller.MPC, Any]:
    mpc = do_mpc.controller.MPC(qc.model)

    setup_mpc = {
        'n_horizon': 30,
        't_step': t_step,   
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
    }
    mpc.set_param(**setup_mpc)

    surpress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
    mpc.set_param(nlpsol_opts = surpress_ipopt)

    mpc_tvp_template = mpc.get_tvp_template()
    def mpc_tvp_fun(t_now):
        return mpc_tvp_template
    mpc.set_tvp_fun(mpc_tvp_fun)


    lterm = 0 
    lterm += sum1((qc.model.x['pos'] -  qc.model.tvp['pos_setpoint'])**2) * qc.model.tvp['setpoint_weight', 0]
    lterm += sum1((qc.model.x['dpos']- qc.model.tvp['dpos_setpoint'])**2) * qc.model.tvp['setpoint_weight', 1]
    lterm += sum1((qc.model.x['omega']- qc.model.tvp['omega_setpoint'])**2) * qc.model.tvp['setpoint_weight', 2]
    lterm += sum1((qc.model.x['phi']- qc.model.tvp['phi_setpoint'])**2) * qc.model.tvp['setpoint_weight', 3]

    mterm = 10*lterm
    mpc.set_objective(lterm=lterm, mterm=mterm)
    mpc.set_rterm(thrust=10)

    mpc.bounds['lower','_u','thrust'] = 0
    mpc.bounds['upper','_u','thrust'] = 0.3

    mpc.setup()

    mpc.x0 = np.ones((qc.model.n_x,1))*1e-3
    mpc.set_initial_guess()

    return mpc, mpc_tvp_template


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

    lqr.set_setpoint(xss = x_ss)
    x0 = simulator.x0

    for k in range(N_iter):
        u0 = lqr.make_step(x0)
        u0 = np.clip(u0, 0, 0.3)
        x0 = simulator.make_step(u0) + np.random.randn(12,1)*v_x


def mpc_flyto(
        simulator: do_mpc.simulator.Simulator, 
        mpc: do_mpc.controller.MPC, 
        tvp_template: Any,
        v_x: Union[np.ndarray, float] = None,
        pos_setpoint: Optional[np.ndarray] = np.ones((3,1)),
        N_iter: Optional[int] = 60,
        ) -> None:

    tvp_template['_tvp',:] = 0 # Reset all setpoints
    tvp_template['_tvp',:, 'pos_setpoint'] = pos_setpoint 
    tvp_template['_tvp',:, 'setpoint_weight', 0] = 10 # [pos, dpos, omega, phi] weights
    tvp_template['_tvp',:, 'setpoint_weight', 1] = 1 # [pos, dpos, omega, phi] weights
    tvp_template['_tvp',:, 'setpoint_weight', 2] = 1 # [pos, dpos, omega, phi] weights
    tvp_template['_tvp',:, 'setpoint_weight', 3] = 1 # [pos, dpos, omega, phi] weights

    if v_x is None:
        v_x = _variance_state_noise

    x0 = simulator.x0
    for k in range(N_iter):
        u0 = mpc.make_step(x0)
        x0 = simulator.make_step(u0) + np.random.randn(12,1)*v_x
        
    
def figure_eight(t, a=1, s=1, height=1):
    """Generate a figure eight trajectory"""
    t = np.atleast_2d(t)
    return a*np.concatenate([
        np.sin(s*t),
        np.sin(s*t)*cos(s*t),
        np.ones_like(s*t)*height
    ], axis=0)

def mpc_figure_eight(
        simulator: do_mpc.simulator.Simulator, 
        mpc: do_mpc.controller.MPC, 
        tvp_template: Any,
        v_x: Union[np.ndarray, float] = None,
        N_iter: Optional[int] = 200,
        ) -> None:

    tvp_template['_tvp',:] = 0 # Reset all setpoints
    tvp_template['_tvp',:, 'setpoint_weight', 0] = 10 # [pos, dpos, omega, phi] weights
    tvp_template['_tvp',:, 'setpoint_weight', 1] = 0.1 # [pos, dpos, omega, phi] weights
    tvp_template['_tvp',:, 'setpoint_weight', 2] = 1 # [pos, dpos, omega, phi] weights
    tvp_template['_tvp',:, 'setpoint_weight', 3] = 0 # [pos, dpos, omega, phi] weights

    if v_x is None:
        v_x = np.array([
            1e-2, 1e-2, 1e-2,
            0,0,0,
            0,0,0,
            0,0,0
        ]).reshape(-1,1)

    x0 = simulator.x0
    for k in range(N_iter):
        t = np.arange(mpc.n_horizon+1)*mpc.t_step+mpc.t0
        tvp_template['_tvp',:, 'pos_setpoint'] =  vertsplit(figure_eight(t, s=1, height=2).T)
        u0 = mpc.make_step(x0)
        x0 = simulator.make_step(u0) + np.random.randn(12,1)*v_x
        


# %% 

if __name__ == '__main__':
    qc = qcmodel.Quadcopter()
    qc.get_model()

    t_step = 0.05
    simulator = get_simulator(t_step, qc)
    lqr = get_LQR(t_step, qc)
    mpc, mpc_tvp_template = get_MPC(t_step, qc)

    # Fly to new position test
    simulator.reset_history()
    mpc.reset_history()
    simulator.x0 = np.zeros((12,1))
    mpc.x0 = simulator.x0
    mpc_flyto(simulator, mpc, mpc_tvp_template)
    fig, ax = plot_results(qc, mpc.data, figsize=(12,8)) 


    # Fly to new position test
    simulator.reset_history()
    mpc.reset_history()
    simulator.x0 = np.zeros((12,1))
    mpc.x0 = simulator.x0
    mpc_figure_eight(simulator, mpc, mpc_tvp_template)
    fig, ax = plot_results(qc, mpc.data, figsize=(12,8)) 

    plt.show(block=True)


# %%
