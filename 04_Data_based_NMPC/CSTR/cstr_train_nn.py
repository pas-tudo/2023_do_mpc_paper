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

class Scaler:
    def __init__(self, data):
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data, idx = None):
        if idx is None:
            mean = self.mean
            std = self.std
        else:
            mean = self.mean[idx]
            std = self.std[idx]

        return (data - mean) / std

    def inverse_transform(self, data, idx = None):
        if idx is None:
            mean = self.mean
            std = self.std
        else:
            mean = self.mean[idx]
            std = self.std[idx]

        if isinstance(data, pd.DataFrame):
            pass
        else:
            mean = mean.to_numpy().reshape(1,-1)
            std = std.to_numpy().reshape(1,-1)

        return data * std + mean

# Train test split
data_train, data_test = train_test_split(data, test_fraction=0.2, random_state=42)

# Scaling
scaler = Scaler(data_train)
data_train_scaled = scaler.transform(data_train)

# Visualization of scaled train data in histogramm
_ = data_train_scaled[['x_k', 'u_k']].hist(sharex=True, sharey=True)

# %%

def get_model(data):

    # Model input
    x_k_tf = keras.Input(shape=(data['x_k'].shape[1]), name='x_k')
    u_k_tf = keras.Input(shape=(data['u_k'].shape[1]), name='u_k')
    layer_in  = keras.layers.concatenate([x_k_tf, u_k_tf])

    # Hidden layers
    for k in range(4):
        layer_in = keras.layers.Dense(32, 
            activation='tanh', 
            name=f'hidden_{k}',
        )(layer_in)

    # Output layer
    dx_next_tf= keras.layers.Dense(data['x_next'].shape[1], name='dx_next_norm')(layer_in)

    x_next_tf = keras.layers.Add(name='x_next_norm')([x_k_tf, dx_next_tf])

    # Create model
    eval_model = keras.Model(inputs=[x_k_tf, u_k_tf], outputs=x_next_tf)
    train_model = keras.Model(inputs=[x_k_tf, u_k_tf], outputs=x_next_tf)

    return train_model, eval_model


# %%
train_model, eval_model = get_model(data_train)


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
    x=[data_train_scaled['x_k'], data_train_scaled['u_k']],
    y=[data_train_scaled['x_next']],
    epochs=1000,
    batch_size=512,
)

# %%

def plot_parity(data, scaler, ax, **kwargs):
    data_scaled = scaler.transform(data)

    x_next_pred_scaled = eval_model.predict([data_scaled['x_k'], data_scaled['u_k']])
    print(x_next_pred_scaled.shape)
    x_next_pred = scaler.inverse_transform(x_next_pred_scaled, idx='x_next')
    x_next_true = data['x_next'].to_numpy()
    for k in range(4):
        i,j = k//2, k%2
        ax[i,j].plot(x_next_true[:,k], x_next_pred[:,k], **kwargs)
        ax[i,j].plot(x_next_true[:,k], x_next_true[:,k], '--', color='k')

# Parity plot for all outputs
fig, ax = plt.subplots(2,2)
plot_parity(data_test, scaler,   ax, alpha=0.5, marker='x', linestyle='None')
plot_parity(data_train, scaler,  ax, alpha=0.5, marker='x', linestyle='None')

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


x = SX.sym('x', 4)
u = SX.sym('u', 2)

x_scaled = scaler.transform(x, idx='x_k')
u_scaled = scaler.transform(u, idx='u_k')

casadi_converter.convert(x_k = x_scaled.T, u_k = u_scaled.T)

x_next_scaled = casadi_converter['x_next_norm']
x_next = scaler.inverse_transform(x_next_scaled, idx='x_next')

cas_function = Function('x_next_fun', [x, u], [x_next])

# %% [markdown]
"""
## Simulate open loop with NN model as CasADi function

"""

# %%
x_open_loop = np.zeros(data['x_k'].shape)

x_open_loop[0] = data['x_k'].iloc[0]

for k, u_k in enumerate(data['u_k'][:-1].to_numpy()):
    x_next = cas_function(x_open_loop[k], u_k)
    x_open_loop[k+1] = x_next

# Plot the results
fig, ax = plt.subplots(3,2, sharex=True)

for k in range(4):
    i,j = k//2, k%2
    ax[i,j].plot(data['x_next'].to_numpy()[:,k], label='True', alpha=0.5)
    ax[i,j].plot(x_open_loop[:,k], label='Open loop', alpha=0.5)

ax[-1,0].plot(data['u_k'].to_numpy()[:,0], label='u_1', alpha=0.5)
ax[-1,1].plot(data['u_k'].to_numpy()[:,1], label='u_2', alpha=0.5)

ax[0,0].set_xlim(0, 100)
