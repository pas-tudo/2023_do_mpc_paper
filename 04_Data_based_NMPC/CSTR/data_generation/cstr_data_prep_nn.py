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

# Control packages
import do_mpc
# %% [markdown]
# # Load closed-loop data
# - Load the sampling plan
# - Initialize the data handler with the sampling plan
# - Load the data from the closed-loop samples

# %%
data_dir = os.path.join('.', 'closed_loop_lqr')

plan = do_mpc.tools.load_pickle(os.path.join(data_dir, 'sampling_plan_lqr.pkl'))

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
    lambda res: res['_x'][:-1, :]
)
dh.set_post_processing('u_k',
    lambda res: res['_u'][:-1, :]
)
dh.set_post_processing('x_next',
    lambda res: res['_x'][1:, :]
)

# %% [markdown]
# Stack the results of all trajectories into a single array and create a `pandas` dataframe.

# %%

X_K = []
U_K = []
X_NEXT = []

for res_i in dh[:]:
    X_K.append(res_i['x_k'])
    U_K.append(res_i['u_k'])
    X_NEXT.append(res_i['x_next'])

X_K = np.concatenate(X_K, axis=0)
U_K = np.concatenate(U_K, axis=0)
X_NEXT = np.concatenate(X_NEXT, axis=0)
dX = X_NEXT - X_K

data = pd.concat(
    [
        pd.DataFrame(X_K, columns=['C_a', 'C_b', 'T_R', 'T_K']),
        pd.DataFrame(U_K, columns=['F', 'Q_dot']),
        pd.DataFrame(X_NEXT, columns=['C_a', 'C_b', 'T_R', 'T_K']),
        pd.DataFrame(dX, columns = ['C_a', 'C_b', 'T_R', 'T_K'])
    ],
    axis=1,
    keys=['x_k', 'u_k', 'x_next', 'dx_next']
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

# %%
ax = data[['dx_next']].hist(figsize=(8,8))
fig = plt.gcf()
fig.tight_layout()


# %% [markdown]
# ## Export data to pickle

# %%
data.to_pickle('cstr_data_lqr.pkl')

# %%
