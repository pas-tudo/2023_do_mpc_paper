from droneplot import DronePlot
from qcmodel import rot_mat
import matplotlib.pyplot as plt
import do_mpc
import pdb


class ResultPlot:
    def __init__(self, quadcopter, data, with_pos_setpoint = True, with_yaw_setpoint = False, **kwargs):
        self.quadcopter = quadcopter
        self.data = data
        self.graphics = do_mpc.graphics.Graphics(data)
            


        self.fig = plt.figure(**kwargs)
        
        self.ax3d = plt.subplot2grid((5,3), (0,0), colspan=2, rowspan=5, projection='3d')

        self.ax = []
        self.ax.append(plt.subplot2grid((5,3), (0,2), colspan=1, rowspan=1))
        self.ax.append(plt.subplot2grid((5,3), (1,2), colspan=1, rowspan=1, sharex=self.ax[0]))
        self.ax.append(plt.subplot2grid((5,3), (2,2), colspan=1, rowspan=1, sharex=self.ax[0]))
        self.ax.append(plt.subplot2grid((5,3), (3,2), colspan=1, rowspan=1, sharex=self.ax[0]))
        self.ax.append(plt.subplot2grid((5,3), (4,2), colspan=1, rowspan=1, sharex=self.ax[0]))

        self.graphics.add_line(var_type='_x', var_name='pos', axis=self.ax[0])
        if with_pos_setpoint:
            self.ax[0].set_prop_cycle(None)
            self.graphics.add_line(var_type='_p', var_name='pos_setpoint', axis=self.ax[0], linestyle='--')  
        self.graphics.add_line(var_type='_x', var_name='dpos', axis=self.ax[1]) 
        self.graphics.add_line(var_type='_x', var_name='phi', axis=self.ax[2])
        if with_yaw_setpoint:
            self.ax[2].set_prop_cycle(None)
            self.graphics.add_line(var_type='_p', var_name='yaw_setpoint', axis=self.ax[2], linestyle='--')
        self.graphics.add_line(var_type='_x', var_name='omega', axis=self.ax[3])
        self.graphics.add_line(var_type='_u', var_name='thrust', axis=self.ax[4])
        
        self.ax[4].set_xlabel('time [s]')
        self.ax[0].set_ylabel('position [m]')
        self.ax[1].set_ylabel('velocity [m/s]')
        self.ax[2].set_ylabel('phi [rad]')
        self.ax[3].set_ylabel('omega [rad/s]')
        self.ax[4].set_ylabel('thrust [N]') 

        self.fig.align_ylabels()
        self.fig.tight_layout()


    def draw(self, k=-1):
        self.graphics.plot_results()
        self.graphics.reset_axes()

        self.ax3d.clear()
        droneplot = DronePlot(self.ax3d, self.quadcopter, scale=10)
        droneplot.draw_at(self.data['_x','pos'][k], rot_mat(*self.data['_x','phi'][k]).full()) 

        self.ax3d.plot(
        self.data['_x','pos',0][:k].flatten(),
        self.data['_x','pos',1][:k].flatten(),
        self.data['_x','pos',2][:k].flatten()
        )

        self.ax3d.axes.set_xlim3d(-1.5, 1.5) 
        self.ax3d.axes.set_ylim3d(-1.5, 1.5)
        self.ax3d.axes.set_zlim3d(bottom=0, top=3) 

        # plt.show()
        plt.pause(0.01)

