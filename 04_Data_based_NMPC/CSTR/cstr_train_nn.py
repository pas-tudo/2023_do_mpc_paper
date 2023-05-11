# %% [markdown]
# # Training a surrogate model of the CSTR with a neural network
# Load packages

# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import tf2onnx
from casadi import *

import os
import sys
from typing import Tuple

import do_mpc

import importlib
importlib.reload(do_mpc.sysid._onnxconversion)

# %% [markdown]
# ## Load data
# The data has been generated with the do-mpc sampling tools in ``./data_generation/cstr_data_generation.py``.
# and was pre-processed in ``./data_generation/cstr_data_prep_nn.py``.
# In this script, we load the data and split it into a training and a test set.

# %%

data = pd.read_pickle(os.path.join('data_generation', 'cstr_data_lqr.pkl'))

# %%

data.head(5)


# %%

def train_test_split(data, test_fraction=0.2, **kwargs):
    data_test = data.sample(frac=test_fraction, **kwargs)
    data_train = data.drop(data_test.index)

    return data_train, data_test

# Train test split
data_train, data_test = train_test_split(data, test_fraction=0.2, random_state=42)
# %% [markdown]
# ## Create a neural network model with keras
# We use the keras functional API to create a neural network model with two inputs (current state and current input) and one output (next state).
# Within the model, we include normalization layers that are adapted to the training data (prior to training).
# The ``get_model`` function returns two models: one for training (scaled outputs) and one for evaluation.
# %%

def get_model(data: pd.DataFrame, n_layer: int = 4, n_neurons: int = 32) -> Tuple[keras.Model, keras.Model, keras.layers.Normalization]:
    """
    Get neural network model architecture.

    Parameters
    ----------
    data : pd.DataFrame
        Data used for training. This data is used to adapt the normalization layers and determine the input and output dimensions.
    n_layer : int, optional
        Number of hidden layers, by default 4
    n_neurons : int, optional
        Number of neurons per hidden layer, by default 32

    Returns
    -------
    train_model : keras.Model
        Model used for training (returns scaled outputs)
    eval_model : keras.Model
        Model used for evaluation (returns unscaled outputs)
    scale_outputs : keras.layers.Normalization
        Normalization layer for the outputs. Should be called on targets before training.
    """


    # Model input
    x_k_tf = keras.Input(shape=(data['x_k'].shape[1]), name='x_k')
    u_k_tf = keras.Input(shape=(data['u_k'].shape[1]), name='u_k')

    # Normalization layers
    scale_input_x = keras.layers.Normalization(name='scale_input_x')
    scale_input_u = keras.layers.Normalization(name='scale_input_u')
    scale_input_x.adapt(data[['x_k']].to_numpy())
    scale_input_u.adapt(data[['u_k']].to_numpy())
    scale_outputs = keras.layers.Normalization(name='x_next_scaled')
    scale_outputs.adapt(data['x_next'].to_numpy())
    unscale_outputs = keras.layers.Normalization(invert=True, name='x_next')
    unscale_outputs.adapt(data['x_next'].to_numpy())

    x_k_scaled = scale_input_x(x_k_tf)
    u_k_scaled = scale_input_u(u_k_tf)

    layer_in  = keras.layers.concatenate([x_k_scaled, u_k_scaled])

    # Hidden layers
    for k in range(n_layer):
        layer_in = keras.layers.Dense(n_neurons, 
            activation='tanh', 
            name=f'hidden_{k}',
        )(layer_in)

    # Output layer
    dx_next_tf_scaled = keras.layers.Dense(data['x_next'].shape[1], name='x_next_norm')(layer_in)
    x_next_tf_scaled = keras.layers.Add(name='x_next_scaled')([x_k_scaled, dx_next_tf_scaled])

    x_next_tf = unscale_outputs(x_next_tf_scaled)

    # Create model
    eval_model = keras.Model(inputs=[x_k_tf, u_k_tf], outputs=x_next_tf)
    train_model = keras.Model(inputs=[x_k_tf, u_k_tf], outputs=x_next_tf_scaled)

    return train_model, eval_model, scale_outputs


# %%
train_model, eval_model, scale_outputs = get_model(data_train, n_layer=4, n_neurons=32)

# %% [markdown]
# ## Prepare model for training and fit model
# The model is trained with the Adam optimizer and the mean squared error loss.
# A summary of the model is shown below and the model is trained for 500 epochs.

# %%
# Prepare model for training
train_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mse'],
)

train_model.summary()

# %%
# Fit model
history = train_model.fit(
    x=[data_train['x_k'], data_train['u_k']],
    y=[scale_outputs(data_train['x_next'])],
    epochs=500,
    batch_size=512,
)

# %% [markdown]
# ## Evaluate model performance
# The model performance is evaluated on the test set by plotting the parity plots for the outputs.

# %%

def plot_parity(data, ax, **kwargs):
    x_next_pred = eval_model.predict([data['x_k'], data['u_k']])
    x_next_true = data['x_next'].to_numpy()
    names = list(data['x_k'].columns)
    for k in range(4):
        i,j = k//2, k%2
        ax[i,j].plot(x_next_true[:,k], x_next_pred[:,k], **kwargs)
        ax[i,j].plot(x_next_true[:,k], x_next_true[:,k], '--', color='k')
        ax[i,j].set_title(names[k])
        ax[i,j].set_xlabel('True')
        ax[i,j].set_ylabel('Predicted')
    ax[0,0].legend()

# Parity plot for all outputs
fig, ax = plt.subplots(2,2)
plot_parity(data_train,ax, alpha=0.5, marker='x', linestyle='None', label='Train')
plot_parity(data_test, ax, alpha=0.5, marker='x', linestyle='None', label='Test')
fig.tight_layout()



# %% [markdown]

# ## Export model to ONNX



# %% [code]
# Export model to ONNX
model_input_signature = [
    tf.TensorSpec(np.array((1, 4)), name='x_k'),
    tf.TensorSpec(np.array((1, 2)), name='u_k'),
]
output_path = os.path.join('models', 'cstr_onnx.onnx')

onnx_model, _ = tf2onnx.convert.from_keras(eval_model, 
    output_path=output_path, 
    input_signature=model_input_signature
)

# %% [markdown]

# ## Test the ONNX conversion
# - Initialize the `ONNXConversion` class with the ONNX model
# - Create CasADi symbolic variables for the state and input
# - Call the `convert` method with the scaled symbolic variables as arguments
# - The `ONNXConversion` class can be indexed to retrieve the symbolic expressions for the next state
# - Create a CasADi function with the symbolic expressions for the next state


# %%
casadi_converter = do_mpc.sysid.ONNXConversion(onnx_model)
casadi_converter

# %%
x = SX.sym('x', 4)
u = SX.sym('u', 2)

casadi_converter.convert(x_k = x.T, u_k = u.T)

x_next = casadi_converter['x_next']
cas_function = Function('x_next_fun', [x, u], [x_next])

# %% [markdown]
# Evaluate the CasADi function with the first data point in the test set.

# %%

x_k = data_test['x_k'].iloc[0].to_numpy()
u_k = data_test['u_k'].iloc[0].to_numpy()

x_next_cas = cas_function(x_k, u_k)

print(f'x_next_cas = {x_next_cas}')

# %% [markdown]
"""
## Simulate open loop with NN model as CasADi function

"""

# %%

data_dir = os.path.join('.', 'data_generation', 'closed_loop_lqr')

plan = do_mpc.tools.load_pickle(os.path.join(data_dir, 'sampling_plan_lqr.pkl'))

dh = do_mpc.sampling.DataHandler(plan)
dh.data_dir = os.path.join(data_dir, '')

idx = 5

res = dh[idx][0]['res']

x_open_loop = np.zeros(res['_x'].shape)
x_open_loop[0] = res['_x'][0]

for k, u_k in enumerate(res['_u'][:-1]):
    x_next = cas_function(x_open_loop[k], u_k)
    x_open_loop[k+1] = x_next

# Plot the results
fig, ax = plt.subplots(3,2, sharex=True, figsize=(10,10))

for k in range(4):
    i,j = k//2, k%2
    ax[i,j].plot(res['_x'][:,k], label='True', alpha=0.3, linewidth=5)
    ax[i,j].set_prop_cycle(None)
    ax[i,j].plot(x_open_loop[:,k], label='Open loop', alpha=1)

ax[-1,0].plot(res['_u'][:,0], label='u_1', alpha=0.5)
ax[-1,1].plot(res['_u'][:,1], label='u_2', alpha=0.5)

ax[0,0].set_xlim(0, 500)

# %%
