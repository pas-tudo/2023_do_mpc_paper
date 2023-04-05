#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
import json
from typing import Union, Optional, List, Callable

# class ClosedLoop:
#     def __init__(self, 
#             controller: Union[do_mpc.controller.LQR, do_mpc.controller.MPC], 
#             simulator: do_mpc.simulator.Simulator, 
#             callbacks: List[Callable[[int], None]]=[],
#         ) -> None:
#         self.controller = controller
#         self.simulator = simulator
#         self.callbacks = callbacks

#     def run(self, x0, n_iterations) -> None:
#         for i in range(n_iterations):
#             u0 =  self.controller.make_step(x0)
#             x0 = self.simulator.make_step(u0)
#             for callback in self.callbacks:
#                 callback(i)

def get_mpc(model, bound_dict):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 20,
        'n_robust': 0,
        'open_loop': 0,
        't_step': 0.005,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 1,
        # Use MA27 linear solver in ipopt for faster calculations:
        # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }

    mpc.set_param(**setup_mpc)

    mpc.set_param(store_full_solution=True)

    mpc.scaling['_x', 'T_R'] = 100
    mpc.scaling['_x', 'T_K'] = 100
    mpc.scaling['_u', 'Q_dot'] = 2000
    mpc.scaling['_u', 'F'] = 100


    mterm = (model._x['C_b'] - model.tvp['C_b_set'])**2 
    lterm = mterm

    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.set_rterm(F=1e-2, Q_dot = 1e-4)


    mpc.bounds['lower', '_x', 'C_a'] = bound_dict['states']['lower']['C_a']
    mpc.bounds['lower', '_x', 'C_b'] = bound_dict['states']['lower']['C_b']
    mpc.bounds['lower', '_x', 'T_R'] = bound_dict['states']['lower']['T_R']
    mpc.bounds['lower', '_x', 'T_K'] = bound_dict['states']['lower']['T_K']

    mpc.bounds['upper', '_x', 'C_a'] = bound_dict['states']['upper']['C_a']
    mpc.bounds['upper', '_x', 'C_b'] = bound_dict['states']['upper']['C_b']
    mpc.bounds['upper', '_x', 'T_K'] = bound_dict['states']['upper']['T_K']

    mpc.bounds['lower', '_u', 'F'] = bound_dict['inputs']['lower']['F']
    mpc.bounds['lower', '_u', 'Q_dot'] = bound_dict['inputs']['lower']['Q_dot']

    mpc.bounds['upper', '_u', 'F'] = bound_dict['inputs']['upper']['F']
    mpc.bounds['upper', '_u', 'Q_dot'] = bound_dict['inputs']['upper']['Q_dot']

    mpc.set_nl_cons('T_R', model._x['T_R'], ub=bound_dict['states']['upper']['T_R'], soft_constraint=True, penalty_term_cons=1e2)

    mpc_tvp_template = mpc.get_tvp_template()
    mpc_tvp_template['_tvp', :, 'C_b_set'] = 0.8

    def tvp_fun(t_now):
        return mpc_tvp_template

    mpc.set_tvp_fun(tvp_fun)


    alpha_var = np.array([1., 1.05, 0.95])
    beta_var = np.array([1., 1.1, 0.9])

    mpc.set_uncertainty_values(alpha = alpha_var, beta = beta_var)

    mpc.setup()

    return mpc, mpc_tvp_template



def get_lqr(model, xss, uss):
    """
    --------------------------------------------------------------------------
    template_lqr: tuning parameters
    --------------------------------------------------------------------------
    """
    t_sample = 0.005
    model_dc = model.discretize(t_sample)
    
    # Initialize the controller
    lqr = do_mpc.controller.LQR(model_dc)
    
    # Initialize parameters
    setup_lqr = {'n_horizon':None, 't_step': t_sample}
    lqr.set_param(**setup_lqr)

    
    # Set objective
    Q = np.diag(np.array([1,1,.1,.1]))
    R = np.diag(np.array([1e-3, 1e-5]))
    Rdelu = np.diag(np.array([1e-2, 1e-4]))
    
    lqr.set_objective(Q=Q, R=R)
    # lqr.set_rterm(delR = Rdelu)
    
    lqr.setup()

    lqr.set_setpoint(xss, uss)
    return lqr


class Clipper:
    def __init__(self, model):
        self.lb_u = model._u(-np.inf)
        self.ub_u = model._u(np.inf)

    def __call__(self, u):
        u_clipped = np.clip(u, self.lb_u.cat.full(), self.ub_u.cat.full())
        return u_clipped


def get_clipper(model, bound_dict):
    clipper = Clipper(model)
    clipper.lb_u['F'] = bound_dict['inputs']['lower']['F']
    clipper.ub_u['F'] = bound_dict['inputs']['upper']['F']
    clipper.lb_u['Q_dot'] = bound_dict['inputs']['lower']['Q_dot']
    clipper.ub_u['Q_dot'] = bound_dict['inputs']['upper']['Q_dot']

    return clipper

