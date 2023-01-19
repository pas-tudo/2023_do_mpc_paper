from droneplot import DronePlot
from qcmodel import rot_mat
import matplotlib.pyplot as plt



def plot_results(quadcopter, res, **kwargs):
    fig = plt.figure(**kwargs)


    ax3d = plt.subplot2grid((5,3), (0,0), colspan=2, rowspan=5, projection='3d')

    ax = []
    ax.append(plt.subplot2grid((5,3), (0,2), colspan=1, rowspan=1))
    ax.append(plt.subplot2grid((5,3), (1,2), colspan=1, rowspan=1, sharex=ax[0]))
    ax.append(plt.subplot2grid((5,3), (2,2), colspan=1, rowspan=1, sharex=ax[0]))
    ax.append(plt.subplot2grid((5,3), (3,2), colspan=1, rowspan=1, sharex=ax[0]))
    ax.append(plt.subplot2grid((5,3), (4,2), colspan=1, rowspan=1, sharex=ax[0]))


    ax[0].plot(res['_time'],res['_x','pos'])
    ax[0].set_prop_cycle(None)
    ax[0].plot(res['_time'],res['_tvp','pos_setpoint'], linestyle='--')
    ax[1].plot(res['_time'],res['_x','dpos'])
    ax[2].plot(res['_time'],res['_x','phi'])
    ax[3].plot(res['_time'],res['_x','omega'])
    ax[4].step(res['_time'],res['_u','thrust'])

    ax[4].set_xlabel('time [s]')
    ax[0].set_ylabel('position [m]')
    ax[1].set_ylabel('velocity [m/s]')
    ax[2].set_ylabel('phi [rad]')
    ax[3].set_ylabel('omega [rad/s]')
    ax[4].set_ylabel('thrust [N]')

    droneplot = DronePlot(ax3d, quadcopter, scale=10)
    droneplot.draw_at(res['_x','pos'][-1], rot_mat(*res['_x','phi'][-1]).full())

    ax3d.plot(
    res['_x','pos',0].flatten(),
    res['_x','pos',1].flatten(),
    res['_x','pos',2].flatten()
    )

    ax3d.axes.set_xlim3d(-1.5, 1.5) 
    ax3d.axes.set_ylim3d(-1.5, 1.5)
    ax3d.axes.set_zlim3d(bottom=0, top=3) 

    fig.tight_layout()

    ax.append(ax3d)

    return fig, ax

