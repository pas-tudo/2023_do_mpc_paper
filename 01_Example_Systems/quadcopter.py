import numpy as np
from casadi import *
import do_mpc

class Quadcopter:
    """Baseline model for a quadcopter.
    The model represents a Crazyflie 2.0 quadcopter. See datasheet:
    https://www.bitcraze.io/documentation/hardware/crazyflie_2_0/crazyflie_2_0-datasheet.pdf

    Inertia, mass and linear thrust taken from:
    https://www.research-collection.ethz.ch/handle/20.500.11850/214143
    """
    def __init__(self):
        # Parameters
        self.J = np.array([
                [16.571710, 0.830806, 0.718277],
                [0.830806, 16.655602, 1.800197],
                [0.718277, 1.800197, 29.261652],
            ])*1e-6 # inertia [kg/m2]

        self.d_t = 0.005964552 # linear thrust to torque relationship
        self.d_t *= np.array([1,-1,-1,1]) # motor spin directions

        self.l_a = 40*1e-3 # Arm length [m] of the quadcopter
        self.d_y = self.l_a*np.sin([np.pi/4, np.pi/4,-np.pi/4,-np.pi/4]) # position of rotors in body frame [m]
        self.d_x = self.l_a*np.sin([np.pi/4,-np.pi/4, np.pi/4,-np.pi/4]) # position of rotors in body frame [m]

        self.D = np.stack([self.d_y, -self.d_x, self.d_t])

        self.m = 28*1e-3 # mass [kg]

        self.g = np.array([0,0,9.81]) # gravity vector [m/s2]

        # Initialize model
        self.model = do_mpc.model.Model('continuous')
        self.pos = self.model.set_variable('_x',  'pos', (3,1)) # Position in inertial frame
        self.dpos = self.model.set_variable('_x',  'dpos', (3,1)) # Velocity in inertial frame
        self.phi = self.model.set_variable('_x',  'phi', (3,1)) # Orientation in inertial frame (yaw, pitch, roll)
        self.omega = self.model.set_variable('_x',  'omega', (3,1)) # Angular velocity in body frame
        self.f = self.model.set_variable('_u',  'thrust',(4,1)) # Thrust of each rotor

        self.pos_setpoint = self.model.set_variable('_tvp', 'pos_setpoint', (3,1))
        self.dpos_setpoint = self.model.set_variable('_tvp',  'dpos_setpoint', (3,1))
        self.phi_setpoint = self.model.set_variable('_tvp',  'phi_setpoint', (3,1))
        self.omega_setpoint = self.model.set_variable('_tvp',  'omega_setpoint', (3,1))
        self.setpoint_weights = self.model.set_variable('_tvp',  'setpoint_weight', (4,1))


    def get_model(self, process_noise=False):
        # Prepare intermediates
        self.F = vertcat(0,0,sum1(self.f)) # total force in body frame
        self.R = rot_mat(self.phi[0],self.phi[1],self.phi[2]) # rotation matrix from body to inertial frame
        self.ddpos = 1/self.m*self.R@self.F - self.g # acceleration in inertial frame 
        # Compute dphi and domega
        dphi = self.model.set_expression('dphi', self.R@self.omega) # angular velocity in inertial frame
        domega = np.linalg.inv(self.J)@(self.D@self.f-cross(self.omega,self.J@self.omega)) # angular acceleration in body frame


        # Set all RHS
        self.model.set_rhs('pos', self.dpos, process_noise=process_noise)
        self.model.set_rhs('dpos',self.ddpos, process_noise=process_noise)
        self.model.set_rhs('phi', dphi, process_noise=process_noise)
        self.model.set_rhs('omega', domega, process_noise=process_noise)

        self.model.setup()


    def stable_point(self, pos_setpoint, p=0, v=0, w=0, tpv=0):
        """Stable point for the quadcopter model given a setpoint_pos and the configured model.
        """

        f =  sum1((self.model.x['pos']-pos_setpoint)**2)
        f += sum1((self.model.x['dpos'])**2)
        f += sum1((self.model.x['phi'])**2)
        f += sum1((self.model.x['omega'])**2)

        lbx = vertcat(self.model.x(-np.inf), self.model.u(0))

        nlp = {'x':vertcat(self.model.x, self.model.u), 'f': f, 'g':self.model._rhs, 'p':vertcat(self.model.p, self.model.v, self.model.w, self.model.tvp)}
        S = nlpsol('S', 'ipopt', nlp)
        r = S(lbg=0,ubg=0, lbx=lbx, p=vertcat(self.model.p(p), self.model.v(v), self.model.w(w)))


        x_lin, u_lin = np.split(r['x'],[self.model.n_x])

        return x_lin, u_lin

class WeightUncertainQuadcopter(Quadcopter):
    """Modified Version of quadcopter model with uncertain mass.
    """
    def __init__(self):
        super().__init__()

    def get_model(self, *args, **kwargs):
        # Compute dphi and domega
        self.mass_factor = self.model.set_variable('_p', 'mass_factor', (1,1))
        self.m = self.m*self.mass_factor

        self.int_offset_pos = self.model.set_variable('_x', 'int_offset_pos', (3,1))
        self.int_offset_dpos = self.model.set_variable('_x', 'int_offset_dpos', (3,1))

        self.offset_pos = self.model.set_expression('offset_pos', self.pos_setpoint-self.pos)
        self.offset_dpos = self.model.set_expression('doffset_pos', self.dpos_setpoint-self.dpos)

        self.model.set_rhs('int_offset_pos', self.offset_pos)
        self.model.set_rhs('int_offset_dpos',self.offset_dpos)

        super().get_model(*args, **kwargs)

class BiasedQuadcopter(Quadcopter):
    """Modified Version of a quadcopter model with a bias in the acceleration and gyro measurements.
    
    """
    def __init__(self):
        super().__init__()

        # Introduce bias
        self.acc_bias  = self.model.set_variable('_p', 'acc_bias', (3,1))
        self.gyro_bias = self.model.set_variable('_p', 'gyro_bias', (3,1))

    def get_model(self, *args, **kwargs):

        # Add bias
        self.ddpos += self.acc_bias
        self.omega += self.gyro_bias

        super().get_model(*args, **kwargs)


class MeasuredBiasedQuadcopter(BiasedQuadcopter):
    """Modified Version of a quadcopter model with a bias in the acceleration and gyro measurements.
    This model can be used for the MHE task.
    
    """
    def __init__(self):
        super().__init__()

    def get_model(self):
        # Introduce measurements
        self.model.set_meas('ddpos', self.ddpos, meas_noise=True)
        self.model.set_meas('omega', self.omega, meas_noise=True)
        self.model.set_meas('thrust', self.f, meas_noise=True)

        super().get_model(process_noise=True)



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