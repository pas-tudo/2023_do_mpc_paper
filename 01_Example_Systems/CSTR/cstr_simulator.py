import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc


def get_simulator(model, t_step=0.005):
    """
    --------------------------------------------------------------------------
    template_optimizer: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {
        'integration_tool': 'cvodes',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': t_step
    }

    simulator.set_param(**params_simulator)

    tvp_num = simulator.get_tvp_template()
    tvp_num['C_b_set'] = 0.8

    def tvp_fun(t_now):
        return tvp_num

    simulator.set_tvp_fun(tvp_fun)

    p_num = simulator.get_p_template()
    p_num['alpha'] = 1.0
    p_num['beta'] = 1.0
    def p_fun(t_now):
        return p_num

    simulator.set_p_fun(p_fun)

    simulator.setup()

    return simulator
