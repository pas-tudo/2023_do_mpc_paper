# %% [markdown]
# # Train approximate MPC controller for quadcopter
# Load packages_

# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from casadi import *

import os
import sys
from typing import Tuple


import importlib

# %%

data = pd.read_pickle(os.path.join('data_generation', 'qc_data_mpc.pkl'))

within_max_dist_setpoint = np.linalg.norm(data['x_k'][['dx0', 'dx1', 'dx2']], axis=1) <= 1

# data = data[within_max_dist_setpoint]

# %%

def train_test_split(data, test_fraction=0.2, **kwargs):
    data_test = data.sample(frac=test_fraction, **kwargs)
    data_train = data.drop(data_test.index)

    return data_train, data_test

# Train test split
data_train, data_test = train_test_split(data, test_fraction=0.2, random_state=42)

# %%
# Sumamry of pandas dataframe
data_train.describe()


# %%

def get_model(
        data: pd.DataFrame, 
        n_layer: int = 4, 
        n_neurons: int = 32, 
        activation='tanh',
        ) -> Tuple[keras.Model, keras.Model, keras.layers.Normalization]:
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
    inputs = keras.Input(shape=(data[['x_k', 'u_k_prev', 'p_k']].shape[1]), name='inputs')
    scale_inputs = keras.layers.Normalization()
    scale_inputs.adapt(data[['x_k', 'u_k_prev', 'p_k']].to_numpy())

    # Out   
    scale_outputs = keras.layers.Normalization()
    scale_outputs.adapt(data['u_k'].to_numpy())
    unscale_outputs = keras.layers.Normalization(invert=True)
    unscale_outputs.adapt(data['u_k'].to_numpy())

    layer_in = scale_inputs(inputs)
    #layer_in = inputs

    # Hidden layers
    for k in range(n_layer):
        layer_in = keras.layers.Dense(n_neurons, 
            activation=activation, 
            name=f'hidden_{k}',
        )(layer_in)

    # Output layer
    u_k_scaled = keras.layers.Dense(data['u_k'].shape[1])(layer_in)

    u_k_tf = unscale_outputs(u_k_scaled)

    # Create model
    eval_model = keras.Model(inputs=inputs, outputs=u_k_tf)
    train_model = keras.Model(inputs=inputs, outputs=u_k_scaled)

    return train_model, eval_model, scale_outputs


# %%
train_model, eval_model, scale_outputs = get_model(data_train, n_layer=6, n_neurons=80, activation='relu')


# Prepare model for training
train_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mse'],
)

train_model.summary()
# %%
# Fit model

early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=100,
    verbose=1,
    mode='auto',
)

history = train_model.fit(
    x=data_train[['x_k', 'u_k_prev', 'p_k']].to_numpy(),
    y=[scale_outputs(data_train['u_k'])],
    validation_data=(
        data_test[['x_k', 'u_k_prev', 'p_k']].to_numpy(),
        [scale_outputs(data_test['u_k'])]
    ),
    epochs=100,
    batch_size=1024,
    callbacks=[early_stopping_callback],
)

# %%

def plot_parity(data, ax, **kwargs):
    pred = eval_model.predict(data[['x_k','u_k_prev','p_k']])
    true = data['u_k'].to_numpy()
    for k in range(4):
        i,j = k//2, k%2
        ax[i,j].plot(true[:,k], pred[:,k], **kwargs)
        ax[i,j].plot(true[:,k], true[:,k], '--', color='k')

# Parity plot for all outputs
fig, ax = plt.subplots(2,2)
plot_parity(data_test, ax, alpha=0.5, marker='x', linestyle='None', label='test')
plot_parity(data_train,ax, alpha=0.5, marker='x', linestyle='None', label='train')
ax[0,0].legend()

# %% [markdown]
"""
## Export model with Keras
"""

# %%
# Save model

eval_model.save(os.path.join('models', 'qc_approx_mpc_model'))
# %%
loaded_model = keras.models.load_model(os.path.join('models', 'qc_approx_mpc_model'))

# %%
test_input = [np.zeros((1,17))]

# %%
loaded_model(test_input).numpy()
# %%
eval_model(test_input).numpy()

# %%

loaded_model.layers[1].invert = False
# %%
loaded_model.layers[-1].invert = True
# %%
