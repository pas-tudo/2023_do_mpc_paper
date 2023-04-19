import numpy as np
from casadi import *


class Trajectory:
    def __init__(self):

        self.t = SX.sym('t')

        self._f_x = DM(0)
        self._f_y = DM(0)

    @property
    def f_x(self):
        return self._f_x

    @f_x.setter
    def f_x(self, f_x: SX):
        self._f_x = f_x
        self._f_x_fun = Function('f_x', [self.t], [self._f_x])
        self._df_x_fun = Function('df_x', [self.t], [jacobian(self._f_x, self.t)])

    @property
    def f_y(self):
        return self._f_y

    @f_y.setter
    def f_y(self, f_y: SX):
        self._f_y = f_y
        self._f_y_fun = Function('f_y', [self.t], [self._f_y])
        self._df_y_fun = Function('df_y', [self.t], [jacobian(self._f_y, self.t)])

    @property
    def f_z(self):
        return self._f_z

    @f_z.setter
    def f_z(self, f_z: SX):
        self._f_z = f_z
        self._f_z_fun = Function('f_z', [self.t], [self._f_z])

    def theta(self, t):
        dx = self._df_x_fun(t)
        dy = self._df_y_fun(t)
        return atan2(dy, dx)

    def __call__(self, t):
        return np.concatenate((self._f_x_fun(t), self._f_y_fun(t), self._f_z_fun(t), self.theta(t)), axis=1)

def get_circle(s: float, a:float, height:float) -> Trajectory:

    tra = Trajectory()

    tra.f_x = a * cos(s*tra.t)
    tra.f_y = a * sin(s*tra.t)
    tra.f_z = cos(s*tra.t)+height

    return tra


def get_figure_eight(s: float, a:float, height:float, rot:float = 0) -> Trajectory:

    tra = Trajectory()

    tra.f_x = a * sin(s*tra.t)
    tra.f_y = a * sin(s*tra.t)*cos(s*tra.t)
    tra.f_z = DM(height)

    f_x_new = (tra.f_x*np.cos(rot) - tra.f_y*np.sin(rot))
    f_y_new = (tra.f_x*np.sin(rot) + tra.f_y*np.cos(rot))    

    tra.f_x = f_x_new
    tra.f_y = f_y_new

    return tra

def get_wobbly_figure_eight(s: float, a:float, height:float, rot:float = 0, wobble:float = 0) -> Trajectory:

    tra = get_figure_eight(s=s, a=a, height=height, rot=rot)

    wobble = min(wobble, height)

    tra.f_z = wobble*cos(s*tra.t)+height

    return tra

if __name__ == '__main__':

    tra_list = []
    tra_list.append(get_figure_eight(1, 1, 1))
    tra_list.append(get_figure_eight(1, 1, 1, rot=np.pi/2))

    t = np.linspace(0,4*np.pi, 100)


    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3,1)

    for tra in tra_list:
        x = tra(t)
        ax[0].plot(x[:,0], x[:,1])
        ax[1].plot(t, x[:,:3])
        ax[2].plot(t, x[:,3])

    plt.show(block=True)





    