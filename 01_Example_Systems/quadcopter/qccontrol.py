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

# Typing information
from typing import Tuple, List, Dict, Union, Optional, Any

# Specialized packages
from casadi import *
from casadi.tools import *

# Control packages
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
def get_simulator(t_step: float, model: do_mpc.model.Model) -> do_mpc.simulator.Simulator:
    """
    Create a simulator for the quadcopter model.
    """
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step = t_step)

    # Pass dummy tvp function (these wont affect the simulation)
    sim_tvp_template = simulator.get_tvp_template()
    simulator.set_tvp_fun(lambda t: sim_tvp_template)

    simulator.setup()

    return simulator
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
    lterm += sum1((model.x['pos'] - model.tvp['pos_setpoint'])**2)# * qc.model.tvp['setpoint_weight', 0]
    lterm += .1*sum1((model.x['dpos'])**2) #* model.tvp['setpoint_weight', 0]
    lterm += .1*sum1((model.x['omega'])**2)# * model.tvp['setpoint_weight', 1]
    lterm += sum1((model.x['phi']- model.tvp['phi_setpoint'])**2) #* model.tvp['setpoint_weight', 3]

    mterm = 10*lterm
    mpc.set_objective(lterm=lterm, mterm=mterm)
    mpc.set_rterm(thrust=10)

    mpc.bounds['lower','_u','thrust'] = 0
    mpc.bounds['upper','_u','thrust'] = 0.3

    mpc.setup()

    mpc.x0 = np.ones((model.n_x,1))*1e-3
    mpc.set_initial_guess()

    return mpc, mpc_tvp_template

def get_aux_LQR(t_step: float, tracking_model: do_mpc.model.LinearModel):

    tracking_model_discrete = tracking_model.discretize(t_step)

    lqr = do_mpc.controller.LQR(tracking_model_discrete)
    lqr.set_param(t_step = t_step)
    

    Q = np.diag(np.array([
        1,1,1, # Position
        0,0,0, # Velocity
        1,1,1, # Angle
    ]))
    R = np.diag(np.array([
        1e-1,1e-1,1e-1,
        1e-2,1e-2,1e-2,
    ]))

    lqr.set_objective(Q = Q, R = R)
    lqr.setup()

    return lqr


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

    tvp = x_ss[:9,:]

    lqr.set_setpoint(xss = x_ss, uss = lqr.uss)
    x0 = simulator.x0

    for k in range(N_iter):
        u0 = lqr.make_step(x0)
        u0 = np.clip(u0, 0, 0.3)
        x0 = simulator.make_step(u0) + np.random.randn(12,1)*v_x

        lqr.data.update(_tvp=tvp)


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
        
