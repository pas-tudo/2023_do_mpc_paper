import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import do_mpc
from typing import Tuple, Optional
from enum import Enum, auto
import pdb

def get_random_uniform_func(bound_dict, var_type, names, reduce_range = 0):
    """
    Generate a function that returns a random uniform value within the bounds of the given variable.
    """

    if isinstance(names, str):
        names = [names]

    lb = np.array([bound_dict[var_type]['lower'][name] for name in names]).reshape(-1, 1)
    ub = np.array([bound_dict[var_type]['upper'][name] for name in names]).reshape(-1, 1)

    # Reduce the range of the bounds by the given factor
    delta = ub - lb
    reduce_range_as_float_between_0_and_1 = np.clip(reduce_range, 0, 1, dtype=np.float64) 
    lb = lb + (delta * reduce_range_as_float_between_0_and_1)/2
    ub = ub - (delta * reduce_range_as_float_between_0_and_1)/2


    def random_uniform():
        return np.random.uniform(lb, ub)
    return random_uniform


def plot_cstr_results_new(
        data: do_mpc.data.Data, 
        fig_ax: Optional[Tuple[mpl.figure.Figure, np.ndarray]] = None,
        with_setpoint: bool = True,
        with_legend: bool = True,
        **kwargs
        ) -> Tuple[mpl.figure.Figure, np.ndarray, do_mpc.graphics.Graphics]:
    """
    Configure the plot for the CSTR results.

    Parameters
    ----------
    data : do_mpc.data.Data
        The data object containing the results.
    fig_ax : Tuple[mpl.figure.Figure, np.ndarray], optional
        The figure and axes to plot on. If None, a new figure is created.
    with_setpoint : bool, optional
        Whether to plot the setpoint for the controlled variable.
    with_legend : bool, optional
        Whether to plot the legend.
    **kwargs:
        Additional keyword arguments are passed to the `plt.plot` function.

    Returns
    -------
    fig : mpl.figure.Figure
        The figure. If `fig_ax` is not None, this is the same as `fig_ax[0]`.
    ax : np.ndarray 
        The axes. If `fig_ax` is not None, this is the same as `fig_ax[1]`.
    graphics : do_mpc.graphics.Graphics
        The graphics object.
    """

    # Initialize graphic:
    graphics = do_mpc.graphics.Graphics(data)

    if fig_ax is None:
        fig, ax = plt.subplots(4,1, sharex=True)
    else:
        fig, ax = fig_ax

    # Configure plot:
    graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0], **kwargs)
    graphics.add_line(var_type='_x', var_name='C_b', axis=ax[0], **kwargs)
    graphics.add_line(var_type='_x', var_name='T_R', axis=ax[1], **kwargs)
    graphics.add_line(var_type='_x', var_name='T_K', axis=ax[1], **kwargs)
    graphics.add_line(var_type='_u', var_name='Q_dot', axis=ax[2], **kwargs)
    graphics.add_line(var_type='_u', var_name='F', axis=ax[3], **kwargs)
    
    if with_setpoint:
        graphics.add_line(var_type='_tvp', var_name='C_b_set', axis=ax[0], **kwargs)

    ax[0].set_ylabel('c [mol/l]')
    ax[1].set_ylabel('T [K]')
    ax[2].set_ylabel('Q [kW]')
    ax[3].set_ylabel('Flow [l/h]')
    ax[3].set_xlabel('time [h]')

    # Update properties for all prediction lines:
    for line_i in graphics.pred_lines.full:
        line_i.set_linewidth(1)

    if with_legend:
        label_lines = graphics.result_lines['_x', 'C_a']+graphics.result_lines['_x', 'C_b']
        ax[0].legend(label_lines, ['C_a', 'C_b'])
        label_lines = graphics.result_lines['_x', 'T_R']+graphics.result_lines['_x', 'T_K']
        ax[1].legend(label_lines, ['T_R', 'T_K'])

    fig.align_ylabels()
    fig.tight_layout()

    return fig, ax, graphics


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