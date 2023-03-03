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

    return simulator

# %%
def get_MPC(t_step: float, CSTR: CSTRmodel.CSTR_Cascade, n_robust: int, n_scenarios: int) -> Tuple[do_mpc.controller.MPC, Any]:
    mpc = do_mpc.controller.MPC(CSTR.model)

    setup_mpc = {
        'n_horizon': 35,
        't_step': t_step,   
        'store_full_solution': True,
        'collocation_deg': 4,
        'collocation_ni':1,
        'n_robust':n_robust,
        # Use MA27 linear solver in ipopt for faster calculations:
        'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
    }
    mpc.set_param(**setup_mpc)

    surpress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
    mpc.set_param(nlpsol_opts = surpress_ipopt)
    #mpc.set_param(nlpsol_opts = {})

    mpc_tvp_template = mpc.get_tvp_template()
    def mpc_tvp_fun(t_now):
        return mpc_tvp_template
    mpc.set_tvp_fun(mpc_tvp_fun)

    mpc_p_template = mpc.get_p_template(n_scenarios)
    def mpc_p_fun(t_now):
        return mpc_p_template
    mpc.set_p_fun(mpc_p_fun)

    # Constraints
    lb_x=0*np.ones((CSTR.nx,1))
    ub_x=4*np.ones((CSTR.nx,1))
    # state constraints
    lb_cA=0
    ub_cA=4
    lb_cB=0
    ub_cB=4
    lb_cR=0
    ub_cR=4
    lb_cS=0
    ub_cS=0.12
    lb_Tr=20
    ub_Tr=80


    lb_x[:CSTR.n_reac]=lb_cA
    ub_x[:CSTR.n_reac]=ub_cA
    lb_x[CSTR.n_reac:2*CSTR.n_reac]=lb_cB
    ub_x[CSTR.n_reac:2*CSTR.n_reac]=ub_cB
    lb_x[2*CSTR.n_reac:3*CSTR.n_reac]=lb_cR
    ub_x[2*CSTR.n_reac:3*CSTR.n_reac]=ub_cR
    lb_x[3*CSTR.n_reac:4*CSTR.n_reac]=lb_cS
    ub_x[3*CSTR.n_reac:4*CSTR.n_reac]=ub_cS
    lb_x[4*CSTR.n_reac:]=lb_Tr
    ub_x[4*CSTR.n_reac:]=ub_Tr
    # input constraints
    lb_uA = 0
    ub_uA = 1.5
    lb_uB = 0
    ub_uB = 1.5
    lb_Tj = 20
    ub_Tj = 80
    lb_u=0*np.ones((CSTR.nu,1))
    ub_u=np.inf*np.ones((CSTR.nu,1))
    lb_u[:CSTR.n_reac]=lb_uA
    ub_u[:CSTR.n_reac]=ub_uA
    lb_u[CSTR.n_reac:2*CSTR.n_reac]=lb_uB
    ub_u[CSTR.n_reac:2*CSTR.n_reac]=ub_uB
    lb_u[2*CSTR.n_reac:]=lb_Tj
    ub_u[2*CSTR.n_reac:]=ub_Tj


    QS = 0.5
    QS = QS*np.diag(np.ones(CSTR.n_reac))
    QS[-1,-1]*CSTR.n_reac
    QA = 0
    QA = QA*np.diag(np.ones(CSTR.n_reac))
    #QA[-1,-1]CSTR.n_reac
    QB = 0
    QB = QB*np.diag(np.ones(CSTR.n_reac))
    #QB[-1,-1]CSTR.n_reac
    QR = 1
    QR = QR*np.diag(np.ones(CSTR.n_reac))
    QR[-1,-1]*CSTR.n_reac
    #print(Q)
    R_c = 0.001
    R_c = R_c*np.diag(np.ones(CSTR.nu))
    for i in range(CSTR.n_reac):
        R_c[-i,-i]=R_c[-i,-i]/(60**2/(1.5**2))

    lterm = 0 
    lterm += CSTR.model.x['cS'].T@QS@CSTR.model.x['cS']+CSTR.model.x['cA'].T@QA@CSTR.model.x['cA']+CSTR.model.x['cB'].T@QB@CSTR.model.x['cB']+(CSTR.model.x['cR']-1.5).T@QR@(CSTR.model.x['cR']-1.5)
    lterm +=(sum1(CSTR.model.u['uA'])-ub_u[0])@(sum1(CSTR.model.u['uA'])-ub_u[0])
    lterm +=(sum1(CSTR.model.u['uB'])-ub_u[CSTR.n_reac])@(sum1(CSTR.model.u['uB'])-ub_u[CSTR.n_reac])
    lterm +=(0.0001/(ub_Tj-lb_Tj)**2)*(CSTR.model.u['Tj']-lb_Tj).T@(CSTR.model.u['Tj']-lb_Tj)

    mterm = DM.zeros((1,1))
    mpc.set_objective(lterm=lterm, mterm=mterm)
    mpc.set_rterm(uA=0.001)
    mpc.set_rterm(uB=0.001)
    mpc.set_rterm(Tj=0.001/(60**2/(1.5**2)))

    mpc.bounds['lower','_u','uA'] = lb_uA
    mpc.bounds['lower','_u','uB'] = lb_uB
    mpc.bounds['lower','_u','Tj'] = lb_Tj
    mpc.bounds['upper','_u','uA'] = ub_uA
    mpc.bounds['upper','_u','uB'] = ub_uB
    mpc.bounds['upper','_u','Tj'] = ub_Tj

    mpc.bounds['lower','_x','cA']=lb_cA
    mpc.bounds['lower','_x','cB']=lb_cB
    mpc.bounds['lower','_x','cR']=lb_cR
    mpc.bounds['lower','_x','cS']=lb_cS
    mpc.bounds['lower','_x','Tr']=lb_Tr
    mpc.bounds['upper','_x','cA']=ub_cA
    mpc.bounds['upper','_x','cB']=ub_cB
    mpc.bounds['upper','_x','cR']=ub_cR
    mpc.bounds['upper','_x','cS']=ub_cS
    mpc.bounds['upper','_x','Tr']=ub_Tr

    mpc.set_nl_cons('u_A',sum1(CSTR.model.u['uA']),ub=ub_uA,soft_constraint=False)
    mpc.set_nl_cons('u_B',sum1(CSTR.model.u['uB']),ub=ub_uB,soft_constraint=False)

    mpc.scaling['_x', 'cA'] = 1.5
    mpc.scaling['_x', 'cB'] = 1.5
    mpc.scaling['_x', 'cR'] = 1.5
    mpc.scaling['_x', 'cS'] = 0.12
    mpc.scaling['_x', 'Tr'] = 80
    mpc.scaling['_u', 'Tj'] = 80
    mpc.scaling['_u', 'uA'] = 1.5
    mpc.scaling['_u', 'uB'] = 1.5

    mpc.setup()
    x_0=0*np.ones((CSTR.nx,1))
    x_0[-CSTR.n_reac:]=CSTR.Tr_in
    mpc.x0 = x_0
    mpc.set_initial_guess()

    return mpc, mpc_tvp_template, mpc_p_template

# %% LQR
def template_lqr(model,t_sample):
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
