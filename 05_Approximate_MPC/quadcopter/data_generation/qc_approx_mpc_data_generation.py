
# %% [markdown]
"""
# Data generation for Quadcopter example

Import the necessary packages.
"""
# %% [code]
# Import packages

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import pandas as pd
import time as time
import importlib
import sys 
import os
import json
import pathlib
import multiprocessing as mp
import importlib

from casadi import *
from casadi.tools import *

# CSTR files
example_path = os.path.join('..','..','..','01_Example_Systems','quadcopter')
sys.path.append(example_path)
import qcmodel
import qccontrol
import plot_results
import qctrajectory

# Control packages
import do_mpc

IS_INTERACTIVE = hasattr(sys, 'ps1')


# --------------------

data_dir = os.path.join('.', 'closed_loop_mpc_02')

# --------------------

# %% [markdown]

# ## Create sampling plan

# %% [code]

np.random.seed(99)

def get_uniform_func(lb, ub):
    def f():
        return np.random.uniform(lb, ub)

    return f


if __name__ ==  '__main__' :
    sp = do_mpc.sampling.SamplingPlanner(overwrite=True)
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    sp.set_sampling_var('phi0',  get_uniform_func(-np.pi/8, np.pi/8))
    sp.set_sampling_var('phi1',  get_uniform_func(-np.pi/8, np.pi/8))
    sp.set_sampling_var('phi2',  get_uniform_func(-np.pi/8, np.pi/8))
    sp.set_sampling_var('speed',  get_uniform_func(0.5, 1.5))
    sp.set_sampling_var('radius',  get_uniform_func(0.5, 1.5))
    sp.set_sampling_var('wobble_height',   get_uniform_func(0.5, 1.5))
    sp.set_sampling_var('rot', get_uniform_func(-np.pi, np.pi))
    sp.set_sampling_var('input_noise_dist', get_uniform_func(1e-3, 1e-2))

    plan = sp.gen_sampling_plan(50)
    sp.export(os.path.join(data_dir, 'sampling_plan_mpc'))

    print(pd.DataFrame(plan))



# %% [markdown]
# ## Create sampling function


# %% [code]
t_step = 0.04
qcconf = qcmodel.QuadcopterConfig()
simulator, sim_p_template = qccontrol.get_simulator(t_step, qcmodel.get_model(qcconf, with_pos=True))
mpc, mpc_p_template = qccontrol.get_MPC(t_step, qcmodel.get_model(qcconf, with_pos=True))
nlp_diff = do_mpc.differentiator.DoMPCDifferentiatior(mpc)
nlp_diff.settings.check_LICQ = False
nlp_diff.settings.check_SC = False
nlp_diff.settings.check_rank = False


class SensitivityData:
    def __init__(self, nlp_diff: do_mpc.differentiator.DoMPCDifferentiator):
        self.nlp_diff = nlp_diff
        self.reset_history()

    def reset_history(self):
        self._du0dx0_arr = []
        self._du0du0_prev_arr = []
        self._licq_arr = []

    def make_step(self):
        self.nlp_diff.differentiate()

        du0dx0 = nlp_diff.sens_num["dxdp", indexf["_u",0,0], indexf["_x0"]]
        du0du_prev = nlp_diff.sens_num["dxdp", indexf["_u",0,0], indexf["_u_prev"]]

        self._du0dx0_arr.append(du0dx0)
        self._du0du0_prev_arr.append(du0du_prev)
        self._licq_arr.append(nlp_diff.status.LICQ)

    @property
    def du0dx0(self):
        return np.stack(self._du0dx0_arr, axis=0)

    @property
    def du0du0_prev(self):
        return np.stack(self._du0du0_prev_arr, axis=0)
    
    @property
    def licq(self):
        return np.array(self._licq_arr).reshape(-1,1)

sens_data = SensitivityData(nlp_diff)

def sampling_function(
        phi0, phi1, phi2, speed, radius, wobble_height, rot, input_noise_dist
    ):

    figure_eight_trajectory = qctrajectory.get_wobbly_figure_eight(
        s=speed, 
        a=radius, 
        height=0, 
        wobble=wobble_height, 
        rot=rot, 
        )

    simulator.reset_history()
    mpc.reset_history()
    sens_data.reset_history()

    simulator.x0['pos'] = figure_eight_trajectory(0).T[:3]
    simulator.x0['phi', 0] = phi0
    simulator.x0['phi', 1] = phi1
    simulator.x0['phi', 2] = phi2

    x0 = simulator.x0.cat.full()

    mpc.x0 = x0

    qccontrol.mpc_fly_trajectory(
        simulator, 
        mpc, 
        sim_p_template, 
        N_iter=200, 
        trajectory=figure_eight_trajectory,
        noise_dist=input_noise_dist,
        callbacks = [sens_data.make_step]
        )

    return {
        'time': mpc.data['_time'],
        'x_k': mpc.data['_x'], 
        'u_k': mpc.data['_u'], 
        'p_k': simulator.data['_p'], 
        'u_k_sim': simulator.data['_u'],
        'du0dx0': sens_data.du0dx0,
        'du0du0_prev': sens_data.du0du0_prev,
        'success': mpc.data['success'], 
        'pos_k': simulator.data['_x', 'pos'],
        'nlp_licq': sens_data.licq,
        }
    
# %% [markdown]
"""
## Sample data

We initialize the ``do_mpc.sampling.Sampler`` object and call the ``sample_data`` method to generate the data.
To initialize the ``Sampler``, we pass:
- The sampling plan
- The sampling function
- A directory where the data will be stored

"""
#%% [code]
if __name__ ==  '__main__' :
    
    sampler = do_mpc.sampling.Sampler(plan, overwrite=True)
    sampler.data_dir = os.path.join(data_dir, '')

    sampler.set_sample_function(sampling_function)

    if IS_INTERACTIVE:
        sampler.sample_idx(0)
    else:
        with mp.Pool(processes=8) as pool:
            p = pool.map(sampler.sample_idx, list(range(sampler.n_samples)))
    # %%

    dh = do_mpc.sampling.DataHandler(plan)
    dh.data_dir = sampler.data_dir

    print(dh[0][0]['res']['du0dx0'].shape)
    print(dh[0][0]['res']['du0du0_prev'].shape)
