import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Tuple, Optional





def plot_cstr_results(
        data, fig_ax: Optional[Tuple[mpl.figure.Figure, np.ndarray]] = None, **kwargs
        ) -> Tuple[mpl.figure.Figure, np.ndarray]:
    """
    Plot the CSTR results

    Parameters
    ----------
    data : do_mpc.data.Data 
        The data object containing the results
    fig_ax : Tuple[mpl.figure.Figure, np.ndarray]
        The figure and axes to plot on (optional)

    Returns
    -------
    fig, ax : Tuple[mpl.figure.Figure, np.ndarray]
    """
    

    if fig_ax is None:
        fig, ax = plt.subplots(6,1, sharex=True)
    else:
        fig, ax = fig_ax


    ax[0].plot(data['_time'], data['_x','C_a'],   linewidth=1, **kwargs)
    ax[1].plot(data['_time'], data['_x','C_b'],   linewidth=1, **kwargs)
    ax[2].plot(data['_time'], data['_x','T_R'],   linewidth=1, **kwargs)
    ax[3].plot(data['_time'], data['_x','T_K'],   linewidth=1, **kwargs)
    ax[4].step(data['_time'], data['_u','F'],     linewidth=1, where='post', **kwargs)
    ax[5].step(data['_time'], data['_u','Q_dot'], linewidth=1, where='post', **kwargs)
    ax[0].set_ylabel('C_a')
    ax[1].set_ylabel('C_b')
    ax[2].set_ylabel('T_R')
    ax[3].set_ylabel('T_K')
    ax[4].set_ylabel('F')
    ax[5].set_ylabel('Q_dot')
    ax[5].set_xlabel('time')

    fig.tight_layout()
    fig.align_ylabels()

    return fig, ax