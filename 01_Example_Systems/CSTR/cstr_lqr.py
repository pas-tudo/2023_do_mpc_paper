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
import do_mpc
import pdb

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


def get_clipper(model):
    clipper = Clipper(model)

    clipper.lb_u['F'] = 5.0
    clipper.ub_u['F'] = 100
    clipper.lb_u['Q_dot'] = -8500
    clipper.ub_u['Q_dot'] = 0

    return clipper

    