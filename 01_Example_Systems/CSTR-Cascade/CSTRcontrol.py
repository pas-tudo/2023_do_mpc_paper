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

# Model
import CSTRmodel
# %%
# Generate simulator for the CSTR model
def get_simulator(t_step: float, CSTR: CSTRmodel.CSTR_Cascade) -> do_mpc.simulator.Simulator:
    """
    Create a simulator for the quadcopter model.
    """
    simulator = do_mpc.simulator.Simulator(CSTR.model)
    simulator.set_param(t_step = t_step)

    # Pass dummy tvp (these wont affect the simulation)
    sim_tvp_template = simulator.get_tvp_template()
    simulator.set_tvp_fun(lambda t: sim_tvp_template)

    sim_p_template = simulator.get_p_template()
    simulator.set_p_fun(lambda t: sim_p_template)

    simulator.setup()

    return simulator, sim_tvp_template, sim_p_template

# %%
def get_MPC(t_step: float, CSTR: CSTRmodel.CSTR_Cascade, n_robust: int, n_scenarios: int, **kwargs) -> Tuple[do_mpc.controller.MPC, Any]:
    mpc = do_mpc.controller.MPC(CSTR.model)

    setup_mpc = {
        'n_horizon': 35,
        't_step': t_step,   
        'store_full_solution': False,
        'collocation_deg': 4,
        'collocation_ni':1,
        'n_robust':n_robust,
        # Use MA27 linear solver in ipopt for faster calculations:
        'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
    }
    setup_mpc.update(kwargs)

    mpc.set_param(**setup_mpc)

    surpress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
    # mpc.set_param(nlpsol_opts = surpress_ipopt)
    #mpc.set_param(nlpsol_opts = {})

    mpc_tvp_template = mpc.get_tvp_template()
    def mpc_tvp_fun(t_now):
        return mpc_tvp_template
    mpc.set_tvp_fun(mpc_tvp_fun)

    mpc_p_template = mpc.get_p_template(n_scenarios)
    def mpc_p_fun(t_now):
        return mpc_p_template
    mpc.set_p_fun(mpc_p_fun)

    lterm = 0 
    lterm += -sum1(CSTR.model.x['cR',-1])
    mterm = lterm

    mpc.set_objective(lterm=lterm, mterm=mterm)
    mpc.set_rterm(uA=1e-3)
    mpc.set_rterm(uB=1e-3)
    mpc.set_rterm(Tj=1e-6)

    mpc.bounds['lower','_u','uA'] = 0
    mpc.bounds['lower','_u','uB'] = 0
    mpc.bounds['lower','_u','Tj'] = 20
    mpc.bounds['upper','_u','Tj'] = 80

    mpc.bounds['lower','_x','cA']= 0
    mpc.bounds['lower','_x','cB']= 0
    mpc.bounds['lower','_x','cR']= 0
    mpc.bounds['lower','_x','cS']= 0 
    mpc.bounds['lower','_x','Tr']= 20
    mpc.bounds['upper','_x','cA']= 1.5
    mpc.bounds['upper','_x','cB']= 1.5
    mpc.bounds['upper','_x','cR']= 1.5
    mpc.bounds['upper','_x','cS']= .12
    mpc.bounds['upper','_x','Tr']= 80

    mpc.set_nl_cons('u_A',sum1(CSTR.model.u['uA']),ub=1.5, soft_constraint=False)
    mpc.set_nl_cons('u_B',sum1(CSTR.model.u['uB']),ub=1.5, soft_constraint=False)

    mpc.scaling['_x', 'cA'] = 1.5
    mpc.scaling['_x', 'cB'] = 1.5
    mpc.scaling['_x', 'cR'] = 1.5
    mpc.scaling['_x', 'cS'] = 0.12
    mpc.scaling['_x', 'Tr'] = 80
    mpc.scaling['_u', 'Tj'] = 80
    mpc.scaling['_u', 'uA'] = 1.5
    mpc.scaling['_u', 'uB'] = 1.5

    mpc.setup()

    return mpc, mpc_tvp_template, mpc_p_template

# %% LQR
def get_lqr(model,t_sample):
    """
    --------------------------------------------------------------------------
    template_lqr: tuning parameters
    --------------------------------------------------------------------------
    """
    model_dc = model.discretize(t_sample)
    
    # Initialize the controller
    lqr = do_mpc.controller.LQR(model_dc)
    
    # Initialize parameters
    setup_lqr = {'n_horizon':None,
              't_step':t_sample}
    lqr.set_param(**setup_lqr)
    
    # Set objective
    Q = np.eye(model.n_x)
    R = 1*np.eye(model.n_u)
    Rdelu = np.eye(model.n_u)
    
    lqr.set_objective(Q=Q, R=R)#, Rdelu=Rdelu)
    
    # set up lqr
    lqr.setup()
    # returns lqr
    return lqr

# %%
def run_closed_loop(controller, simulator, n_steps):
    """
    Run a closed-loop simulation with controller and simulator.

    It is necessary to previously set the ``x0`` of the simulator

    Parameters
    ---------- 
    controller : do_mpc.controller.MPC or do_mpc.controller.LQR
    simulator : do_mpc.simulator.Simulator
    n_steps : int
        Number of steps to simulate.
    """
    x0 = simulator.x0
    for k in range(n_steps):
        u0 = controller.make_step(x0)
        x0 = DM(simulator.make_step(u0))