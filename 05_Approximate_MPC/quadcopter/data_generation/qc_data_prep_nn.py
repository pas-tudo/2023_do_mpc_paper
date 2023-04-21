# %% [markdown]
# # Create data set
# 
# Load data from the generated closed-loop samples and prepare for 

# %%
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
from functools import partial
import pdb
import time

# Control packages
import do_mpc
# %% [markdown]
# # Load closed-loop data
# - Load the sampling plan
# - Initialize the data handler with the sampling plan
# - Load the data from the closed-loop samples

# %%
data_dir = os.path.join('.', 'closed_loop_mpc')

plan = do_mpc.tools.load_pickle(os.path.join(data_dir, 'sampling_plan_mpc.pkl'))

dh = do_mpc.sampling.DataHandler(plan)
dh.data_dir = os.path.join(data_dir, '')

# %% [markdown]
# ## Prepare data for NN training
# We seek to train a model of the type:
# $$x_{k+1} = f(x_k, u_k)$$
# For all trajectories in the data set, we extract the following data:
# - $x_k$: The state at time $k$
# - $u_k$: The control input at time $k$
# - $x_{k+1}$: The state at time $k+1$
# 
# We use the `DataHandler` to post-process the data and extract the data in the desired format.

# %%

dh.set_post_processing('x_k',
    lambda res: res['sim']['_x']
)
dh.set_post_processing('pos_setpoint',
    lambda res: res['sim']['_p', 'pos_setpoint'])
dh.set_post_processing('yaw_setpoint',
    lambda res: res['sim']['_p', 'yaw_setpoint'])
dh.set_post_processing('u_k',
    lambda res: res['sim']['_u']
)


# %% [markdown]
# Stack the results of all trajectories into a single array and create a `pandas` dataframe.

if True:
    plt.ion()
    fig, ax = plt.subplots()

    lines = []

    for dh_i in dh[:]:
        for i, line in enumerate(lines):
            line[0].set_alpha(1/(i+1))

        lines.append(ax.plot(dh_i['x_k'][:,0], dh_i['x_k'][:,1], color='k'))
        plt.show()
        plt.pause(0.5)





# %%
if True:
    X_K = []
    U_K = []
    P_k = []

    for res_i in dh[:]:
        x_k = res_i['x_k']
        x_k[:,:3] = x_k[:,:3] - res_i['pos_setpoint']


        X_K.append(x_k)
        U_K.append(res_i['u_k'])
        P_k.append(res_i['yaw_setpoint'])

    X_K = np.concatenate(X_K, axis=0)
    U_K = np.concatenate(U_K, axis=0)
    P_K = np.concatenate(P_k, axis=0)

    data = pd.concat(
        [
            pd.DataFrame(X_K, columns=['dx0', 'dx1', 'dx2', 'v0', 'v1', 'v2', 'phi0', 'phi1', 'phi2', 'omega0', 'omega1', 'omega2']),
            pd.DataFrame(U_K, columns=['f0', 'f1', 'f2', 'f3']),
            pd.DataFrame(P_K, columns=['yaw_set']),
        ],
        axis=1,
        keys=['x_k', 'u_k', 'p_k']
    )

    # %% [markdown]
    # ## Analsis of the data
    # ### Description:

    # %% [code]

    data.describe()

    # %% [markdown]
    # ### Histograms

    # %%

    ax = data[['x_k', 'u_k']].hist(figsize=(8,8))
    fig = plt.gcf()
    fig.tight_layout()


    # %% [markdown]
    # ## Export data to pickle

    # %%
    data.to_pickle('qc_data_mpc.pkl')

    # %%
