import numpy as np
from casadi import *
import do_mpc


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


def template_model(*args, **kwargs):
    """Template model for a quadcopter.
    The model represents a Crazyflie 2.0 quadcopter. See datasheet:
    https://www.bitcraze.io/documentation/hardware/crazyflie_2_0/crazyflie_2_0-datasheet.pdf

    Inertia, mass and linear thrust taken from:
    https://www.research-collection.ethz.ch/handle/20.500.11850/214143

    Returns
    -------
    model : Configured do_mpc model object.
    """
    # Parameters
    J = np.array([
    [16.571710, 0.830806, 0.718277],
    [0.830806, 16.655602, 1.800197],
    [0.718277, 1.800197, 29.261652],
    ])*1e-6 # inertia [kg/m2]

    d_t = 0.005964552 # linear thrust to torque relationship

    d = 40*1e-3
    d_y = d*np.sin([np.pi/8,np.pi/8,-np.pi/8,-np.pi/8]) # position of rotors in body frame [m]
    d_x = d*np.array([np.pi/8,-np.pi/8,np.pi/8,-np.pi/8])*1e-3 # position of rotors in body frame [m]

    D = np.stack([d_y, d_x, d_t*np.array([1,-1,1,-1])])

    m = 28*1e-3 # mass [g]

    g = np.array([0,0,9.81]) # gravity vector [m/s2]

    model = do_mpc.model.Model('continuous')

    pos = model.set_variable('_x',  'pos', (3,1))
    dpos = model.set_variable('_x',  'dpos', (3,1))
    phi = model.set_variable('_x',  'phi', (3,1)) # yaw, pitch, roll
    omega = model.set_variable('_x',  'omega', (3,1))


    f = model.set_variable('_u',  'thrust',(4,1))
    F = vertcat(0,0,sum1(f))

    R = rot_mat(phi[0],phi[1],phi[2])

    ddpos = 1/m*R@F - g


    dphi = R@omega

    domega = np.linalg.inv(J)@(D@f-cross(omega,J@omega))

    model.set_rhs('pos', dpos)
    model.set_rhs('dpos',ddpos)
    model.set_rhs('phi', dphi)
    model.set_rhs('omega', domega)


    model.setup()

    return model


def stable_point(model, pos_setpoint):
    """Stable point for the quadcopter model given a setpoint_pos and the configured model.
    """

    f =  sum1((model.x['pos']-pos_setpoint)**2)
    f += sum1((model.x['dpos'])**2)
    f += sum1((model.x['phi'])**2)
    f += sum1((model.x['omega'])**2)

    lbx = vertcat(model.x(-np.inf), model.u(0))

    nlp = {'x':vertcat(model.x, model.u), 'f': f, 'g':model._rhs}
    S = nlpsol('S', 'ipopt', nlp)

    r = S(lbg=0,ubg=0, lbx=lbx)
    x_lin, u_lin = np.split(r['x'],[model.n_x])
    

    return x_lin, u_lin