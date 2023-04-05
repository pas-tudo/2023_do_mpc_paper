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


def get_model():
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, 'SX')

    # Certain parameters
    K0_ab = 1.287e12 # K0 [h^-1]
    K0_bc = 1.287e12 # K0 [h^-1]
    K0_ad = 9.043e9 # K0 [l/mol.h]
    R_gas = 8.3144621e-3 # Universal gas constant
    E_A_ab = 9758.3*1.00 #* R_gas# [kj/mol]
    E_A_bc = 9758.3*1.00 #* R_gas# [kj/mol]
    E_A_ad = 8560.0*1.0 #* R_gas# [kj/mol]
    H_R_ab = 4.2 # [kj/mol A]
    H_R_bc = -11.0 # [kj/mol B] Exothermic
    H_R_ad = -41.85 # [kj/mol A] Exothermic
    Rou = 0.9342 # Density [kg/l]
    Cp = 3.01 # Specific Heat capacity [kj/Kg.K]
    Cp_k = 2.0 # Coolant heat capacity [kj/kg.k]
    A_R = 0.215 # Area of reactor wall [m^2]
    V_R = 10.01 #0.01 # Volume of reactor [l]
    m_k = 5.0 # Coolant mass[kg]
    T_in = 130.0 # Temp of inflow [Celsius]
    K_w = 4032.0 # [kj/h.m^2.K]
    C_A0 = (5.7+4.5)/2.0*1.0 # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]

    # States struct (optimization variables):
    C_a = model.set_variable(var_type='_x', var_name='C_a', shape=(1,1))
    C_b = model.set_variable(var_type='_x', var_name='C_b', shape=(1,1))
    T_R = model.set_variable(var_type='_x', var_name='T_R', shape=(1,1))
    T_K = model.set_variable(var_type='_x', var_name='T_K', shape=(1,1))

    # TVP
    C_b_set = model.set_variable(var_type='_tvp', var_name='C_b_set', shape=(1,1))

    # Input struct (optimization variables):
    F = model.set_variable(var_type='_u', var_name='F')
    Q_dot = model.set_variable(var_type='_u', var_name='Q_dot')

    # Fixed parameters:
    alpha = model.set_variable(var_type='_p', var_name='alpha')
    beta = model.set_variable(var_type='_p', var_name='beta')

    # Set expression. These can be used in the cost function, as non-linear constraints
    # or just to monitor another output.
    T_dif = model.set_expression(expr_name='T_dif', expr=T_R-T_K)

    # Expressions can also be formed without beeing explicitly added to the model.
    # The main difference is that they will not be monitored and can only be used within the current file.
    K_1 = beta * K0_ab * exp((-E_A_ab)/((T_R+273.15)))
    K_2 =  K0_bc * exp((-E_A_bc)/((T_R+273.15)))
    K_3 = K0_ad * exp((-alpha*E_A_ad)/((T_R+273.15)))

    # Differential equations
    model.set_rhs('C_a', F*(C_A0 - C_a) -K_1*C_a - K_3*(C_a**2))
    model.set_rhs('C_b', -F*C_b + K_1*C_a - K_2*C_b)
    model.set_rhs('T_R', ((K_1*C_a*H_R_ab + K_2*C_b*H_R_bc + K_3*(C_a**2)*H_R_ad)/(-Rou*Cp)) + F*(T_in-T_R) +(((K_w*A_R)*(-T_dif))/(Rou*Cp*V_R)))
    model.set_rhs('T_K', (Q_dot + K_w*A_R*(T_dif))/(m_k*Cp_k))

    # Build the model
    model.setup()


    return model

def get_steady_state(model, C_b_set):
    """ Get the steady state of the model for a given C_b_setpoint. 
    
    Parameters
    ----------
    model : do_mpc.model.Model
        An instance of the do_mpc.model.Model class representing the CSTR
    C_b_set : float
        The setpoint for the concentration of B in the reactor.


    """
    
    opt_x = struct_symSX([
        entry('_x', sym=model._x),
        entry('_u', sym=model._u),
    ])
    opt_p = struct_symSX([
        entry('_tvp', sym=model._tvp),
        entry('_p', sym=model._p),
    ])

    f = (opt_x['_x', 'C_b'] - opt_p['_tvp', 'C_b_set'])**2

    nlp = {'x':opt_x, 'p': opt_p, 'f': f, 'g':model._rhs}

    S = nlpsol('S', 'ipopt', nlp, {'ipopt.print_level':0 , 'ipopt.sb': 'yes', 'print_time':0})

    p_num = opt_p(0)
    p_num['_tvp', 'C_b_set'] = C_b_set
    p_num['_p', 'alpha'] = 1.0
    p_num['_p', 'beta'] = 1.0

    lb_x = opt_x(0)
    lb_x['_x', 'C_a'] = 0.1
    lb_x['_x', 'C_b'] = 0.1
    lb_x['_x', 'T_R'] = 50
    lb_x['_x', 'T_K'] = 50
    lb_x['_u', 'F'] = 5
    lb_x['_u', 'Q_dot'] = -8500

    ub_x = opt_x(0)
    ub_x['_x', 'C_a'] = 2
    ub_x['_x', 'C_b'] = 2
    ub_x['_x', 'T_R'] = 140
    ub_x['_x', 'T_K'] = 140
    ub_x['_u', 'F'] = 100
    ub_x['_u', 'Q_dot'] = 0

    x0 = lb_x.cat + 0.5*(ub_x.cat-lb_x.cat)

    r = S(lbg=0, ubg=0, p=p_num, lbx=lb_x, ubx=ub_x, x0=x0)

    solver_stats = S.stats()

    if not solver_stats['success']:
        print('Solver failed to find a solution. Steady-state might not exist.')

    opt_x_num = opt_x(r['x'])

    x_ss = opt_x_num['_x'].full()
    u_ss = opt_x_num['_u'].full()

    return x_ss, u_ss, p_num

def get_linear_model(C_b_set = 0.6):
    model = get_model()
    x_ss, u_ss, p_num = get_steady_state(model, C_b_set=C_b_set)

    linear_model = do_mpc.model.linearize(
        model = model,
        xss=x_ss,
        uss=u_ss,
        p0=p_num['_p'],
        tvp0=p_num['_tvp'],
    )

    return linear_model, x_ss, u_ss


if __name__ == '__main__':
    linear_model, x_ss, u_ss = get_linear_model()
