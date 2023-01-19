from .basic import Sphere, Line, Arrow

import numpy as np

class DronePlot:
    '''
    Draws a quadrotor at a given position, with a given attitude.
    '''

    def __init__(self, ax, quadcopter, scale = 1):
        '''
        Initialize the drone plotting parameters.

        Params:
            ax: (matplotlib axis) the axis where the sphere should be drawn
            quadcopter: Instance of quadcopter class
            scale: (float) scale of the quadrotor (increase size for better visability)

        Returns:
            None
        '''

        self.ax = ax

        d = quadcopter.l_a * scale 
        motor_size = d/10
        body_size = d/5

        # Unit vector
        self.b1 = np.array([1.0, 0.0, 0.0]).T * d*1.2
        self.b2 = np.array([0.0, 1.0, 0.0]).T * d*1.2
        self.b3 = np.array([0.0, 0.0, 1.0]).T * d*1.2


        # Center of the quadrotor
        self.body = Sphere(self.ax, body_size, 'y')

        # Arrows for the each body axis
        self.arrow_b1 = Arrow(ax, self.b1, 'r')
        self.arrow_b2 = Arrow(ax, self.b2, 'g')
        self.arrow_b3 = Arrow(ax, self.b3, 'b')

        # Create the arms with motors
        x0 = np.zeros(3)
        motor_pos = np.stack([quadcopter.d_x * scale , quadcopter.d_y * scale, np.zeros(4)]).T

        self.m1 = motor_pos[0]
        self.m2 = motor_pos[1]
        self.m3 = motor_pos[2]
        self.m4 = motor_pos[3]

        self.motor1 = Sphere(self.ax, motor_size, x0=self.m1, c='r')
        self.motor2 = Sphere(self.ax, motor_size, x0=self.m2, c='r')
        self.motor3 = Sphere(self.ax, motor_size, x0=self.m3, c='g')
        self.motor4 = Sphere(self.ax, motor_size, x0=self.m4, c='g')

        self.arm_b1 = Line(self.ax, x0, x0 + self.m1, 'r')
        self.arm_b2 = Line(self.ax, x0, x0 + self.m2, 'r')
        self.arm_b3 = Line(self.ax, x0, x0 + self.m3, 'g')
        self.arm_b4 = Line(self.ax, x0, x0 + self.m4, 'g')


    def draw_at(self, x=np.array([0.0, 0.0, 0.0]).T, R=np.eye(3)):
        '''
        Draw the quadrotor at a given position, with a given direction

        Args:
            x: (3x1 numpy.ndarray) position of the center of the quadrotor, 
                default = [0.0, 0.0, 0.0]
            R: (3x3 numpy.ndarray) attitude of the quadrotor in SO(3)
                default = eye(3)
        
        Returns:
            None
        '''

        # First, clear the axis of all the previous plots
        self.ax.clear()

        # Center of the quadrotor
        self.body.draw_at(x)

        # Each motor
        self.motor1.draw_at(x + R.dot(self.m1))
        self.motor2.draw_at(x + R.dot(self.m2))
        self.motor3.draw_at(x + R.dot(self.m3))
        self.motor4.draw_at(x + R.dot(self.m4))

        # Arrows for the each body axis
        self.arrow_b1.draw_from_to(x, R.dot(self.b1))
        self.arrow_b2.draw_from_to(x, R.dot(self.b2))
        self.arrow_b3.draw_from_to(x, R.dot(self.b3))

        # Quadrotor arms
        self.arm_b1.draw_from_to(x, x + R.dot(self.m1))
        self.arm_b2.draw_from_to(x, x + R.dot(self.m2))
        self.arm_b3.draw_from_to(x, x + R.dot(self.m3))
        self.arm_b4.draw_from_to(x, x + R.dot(self.m4))

