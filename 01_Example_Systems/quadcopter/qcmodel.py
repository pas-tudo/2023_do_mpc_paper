import numpy as np
from casadi import *
import sys
import do_mpc
from dataclasses import dataclass


@dataclass
class QuadcopterConfig:
    J = np.array([
            [16.571710, 0.830806, 0.718277],
            [0.830806, 16.655602, 1.800197],
            [0.718277, 1.800197, 29.261652],
        ])*1e-6 
    """
    Matrix of inertia of the quadcopter in the body frame [kg/m2].
    """

    d_t_i = 0.005964552
    """
    Linear thrust to torque relationship.
    """

    motor_spin_coeff = np.array([1,-1,-1,1])
    """
    Coefficient to account for the direction of the motor spin.
    """

    l_a = 40*1e-3
    """
    Arm length [m] of the quadcopter.
    """

    m = 28*1e-3 
    """
    Mass of the quadcopter [kg].
    """

    g = np.array([0,0,9.81])
    """
    Gravity vector [m/s2].
    """

    @property
    def d_t(self):
        """
        Linear thrust to torque relationship for each rotor (accounting for motor spin direction)
        """
        return self.d_t_i * self.motor_spin_coeff
    
    @property
    def d_y(self):
        """
        y-position of rotors in body frame [m]
        """
        return self.l_a*np.sin([np.pi/4, np.pi/4,-np.pi/4,-np.pi/4])
    
    @property
    def d_x(self):
        """
        x-position of rotors in body frame [m]
        """
        return self.l_a*np.sin([np.pi/4,-np.pi/4, np.pi/4,-np.pi/4])

    @property
    def D(self):
        """
        Stacked vector of :py:attr:`d_x`, :py:attr:`d_y` and :py:attr:`d_t`.
        """
        return np.stack([self.d_y, -self.d_x, self.d_t])


def get_model(conf: QuadcopterConfig, with_pos:bool = True, process_noise:bool=False):
    """Baseline model for a quadcopter.
    The model represents a Crazyflie 2.0 quadcopter. See datasheet:
    https://www.bitcraze.io/documentation/hardware/crazyflie_2_0/crazyflie_2_0-datasheet.pdf

    Inertia, mass and linear thrust taken from:
    https://www.research-collection.ethz.ch/handle/20.500.11850/214143
    """
    with_pos = with_pos

    # Initialize model
    model = do_mpc.model.Model('continuous')
    if with_pos:
        pos = model.set_variable('_x',  'pos', (3,1)) # Position in inertial frame
        pos_setpoint = model.set_variable('_tvp', 'pos_setpoint', (3,1))

    dpos = model.set_variable('_x',  'dpos', (3,1)) # Velocity in inertial frame
    phi = model.set_variable('_x',  'phi', (3,1)) # Orientation in inertial frame (yaw, pitch, roll)
    omega = model.set_variable('_x',  'omega', (3,1)) # Angular velocity in body frame
    f = model.set_variable('_u',  'thrust',(4,1)) # Thrust of each rotor

    # Setpoint variables (used in tracking MPC)
    dpos_setpoint = model.set_variable('_tvp',  'dpos_setpoint', (3,1))
    phi_setpoint = model.set_variable('_tvp',  'phi_setpoint', (3,1))
    # omega_setpoint = model.set_variable('_tvp',  'omega_setpoint', (3,1))
    # setpoint_weights = model.set_variable('_tvp',  'setpoint_weight', (3,1))

    # Prepare intermediates
    F = vertcat(0,0,sum1(f)) # total force in body frame
    R = rot_mat(phi[0],phi[1],phi[2]) # rotation matrix from body to inertial frame
    ddpos = 1/conf.m*R@F - conf.g # acceleration in inertial frame 
    # Compute dphi and domega
    dphi = model.set_expression('dphi', R@omega) # angular velocity in inertial frame
    domega = np.linalg.inv(conf.J)@(conf.D@f-cross(omega,conf.J@omega)) # angular acceleration in body frame


    # Set all RHS
    if with_pos:
        model.set_rhs('pos', dpos, process_noise=process_noise)
    model.set_rhs('dpos',ddpos, process_noise=process_noise)
    model.set_rhs('phi', dphi, process_noise=process_noise)
    model.set_rhs('omega', domega, process_noise=process_noise)

    model.setup()

    return model


def get_tracking_model():

    tracking_model = do_mpc.model.LinearModel('continuous')

    pos = tracking_model.set_variable('_x', 'pos', (3,1))
    dpos = tracking_model.set_variable('_x', 'dpos', (3,1))
    phi = tracking_model.set_variable('_x', 'phi', (3,1))

    dpos_set = tracking_model.set_variable('_u', 'dpos_set', (3,1))
    phi_set = tracking_model.set_variable('_u', 'phi_set', (3,1))

    tau1 = .5
    tau2 = .5

    tracking_model.set_rhs('pos', dpos)
    tracking_model.set_rhs('dpos', (dpos_set-dpos)/tau1)
    tracking_model.set_rhs('phi', (phi_set-phi)/tau2)

    tracking_model.setup()

    return tracking_model


def get_stable_point(model, p=0, v=0, w=0, tvp=0):
    """Stable point for the quadcopter model given a setpoint_pos and the configured model.
    """

    # f =  sum1((self.model.x['pos']-pos_setpoint)**2)
    f = sum1((model.x['dpos'])**2)
    f += sum1((model.x['phi'])**2)
    f += sum1((model.x['omega'])**2)

    lbx = vertcat(model.x(-np.inf), model.u(0))

    nlp = {'x':vertcat(model.x, model.u), 'f': f, 'g':model._rhs, 'p':vertcat(model.p, model.v, model.w, model.tvp)}
    opts = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
    S = nlpsol('S', 'ipopt', nlp, opts)
    r = S(lbg=0,ubg=0, lbx=lbx, p=vertcat(model.p(p), model.v(v), model.w(w), model.tvp(tvp)))


    x_lin, u_lin = np.split(r['x'],[model.n_x])

    return x_lin, u_lin



# Some helper functions

def rot_mat(alpha,beta, gamma):
    """Auxiliary function to compute the rotation matrix.
    
    Parameters
    ----------
    alpha : casadi.DM or casadi.MX or casadi.SX or numpy.ndarray
        Rotation angle around the x-axis (yaw).
    beta : casadi.DM or casadi.MX or casadi.SX or numpy.ndarray
        Rotation angle around the y-axis (pitch).
    gamma : casadi.DM or casadi.MX or casadi.SX or numpy.ndarray
        Rotation angle around the z-axis (roll).

    Returns
    -------
    R : casadi.DM or casadi.MX or casadi.SX
    """

    R = vertcat(
        horzcat(
            cos(alpha)*cos(beta),
            cos(alpha)*sin(beta)*sin(gamma)-sin(alpha)*cos(gamma), 
            cos(alpha)*sin(beta)*cos(gamma)+sin(alpha)*sin(gamma),
        ),
        horzcat(
            sin(alpha)*cos(beta),
            sin(alpha)*sin(beta)*sin(gamma)+cos(alpha)*cos(gamma), 
            sin(alpha)*sin(beta)*cos(gamma)-cos(alpha)*sin(gamma),
        ),
        horzcat(
            -sin(beta),
            cos(beta)*sin(gamma),
            cos(beta)*cos(gamma)
        ),
    )
    return R


def print_progress(k,N, bar_len = 50):
    k = int(max(min(k,N),0))
    percent_done = round(100*(k)/(N-1))

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done
    done_str = '█'*done
    togo_str = '░'*togo

    msg = f"\t Progress: [{done_str}{togo_str}] {percent_done}% done"
    print(msg, end='\r')