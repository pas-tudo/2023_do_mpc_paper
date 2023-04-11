# %% [markdown]
# # Training a surrogate model of the CSTR with a neural network
# Load packages_

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

import do_mpc

import importlib
importlib.reload(do_mpc.sysid._onnxconversion)

# %%

data = pd.read_pickle(os.path.join('data_generation', 'cstr_data_lqr.pkl'))


# %%

def train_test_split(data, test_fraction=0.2, **kwargs):
    data_test = data.sample(frac=test_fraction, **kwargs)
    data_train = data.drop(data_test.index)

    return data_train, data_test

# Train test split
data_train, data_test = train_test_split(data, test_fraction=0.2, random_state=42)
# %%

def get_model(data):
    # Model input
    x_k_tf = keras.Input(shape=(data['x_k'].shape[1]), name='x_k')
    u_k_tf = keras.Input(shape=(data['u_k'].shape[1]), name='u_k')

    # Normalization layers
    scale_inputs = keras.layers.Normalization(name='input_scaled')
    scale_inputs.adapt(data[['x_k', 'u_k']].to_numpy())
    scale_outputs = keras.layers.Normalization(name='x_next_scaled')
    scale_outputs.adapt(data['x_next'].to_numpy())
    unscale_outputs = keras.layers.Normalization(invert=True, name='x_next')
    unscale_outputs.adapt(data['x_next'].to_numpy())

    layer_in  = keras.layers.concatenate([x_k_tf, u_k_tf])
    layer_in = scale_inputs(layer_in)

    # Hidden layers
    for k in range(4):
        layer_in = keras.layers.Dense(32, 
            activation='tanh', 
            name=f'hidden_{k}',
        )(layer_in)

    # Output layer
    x_next_tf_scaled = keras.layers.Dense(data['x_next'].shape[1], name='x_next_norm')(layer_in)

    x_next_tf = unscale_outputs(x_next_tf_scaled)

    # Create model
    eval_model = keras.Model(inputs=[x_k_tf, u_k_tf], outputs=x_next_tf)
    train_model = keras.Model(inputs=[x_k_tf, u_k_tf], outputs=x_next_tf_scaled)

    return train_model, eval_model, scale_outputs


# %%
train_model, eval_model, scale_outputs = get_model(data_train)


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

# %%

def plot_parity(data, ax, **kwargs):
    x_next_pred = eval_model.predict([data['x_k'], data['u_k']])
    x_next_true = data['x_next'].to_numpy()
    for k in range(4):
        i,j = k//2, k%2
        ax[i,j].plot(x_next_true[:,k], x_next_pred[:,k], **kwargs)
        ax[i,j].plot(x_next_true[:,k], x_next_true[:,k], '--', color='k')

# Parity plot for all outputs
fig, ax = plt.subplots(2,2)
plot_parity(data_test, ax, alpha=0.5, marker='x', linestyle='None')
plot_parity(data_train,ax, alpha=0.5, marker='x', linestyle='None')

# %% [markdown]
"""
## Export model to ONNX
"""


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
"""
## Convert ONNX model to CasADi
- Initialize the `ONNXConversion` class with the ONNX model
- Create CasADi symbolic variables for the state and input
- Scale the symbolic variables
- Call the `convert` method with the scaled symbolic variables as arguments
- The `ONNXConversion` class can be indexed to retrieve the symbolic expressions for the next state
- Unscale the symbolic expressions for the next state
- Create a CasADi function with the symbolic expressions for the next state
"""

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
