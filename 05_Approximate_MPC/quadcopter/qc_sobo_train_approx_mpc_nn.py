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

sys.path.append(os.path.join('..', '..' , '01_Example_Systems', 'quadcopter'))

import qccontrol
import qcmodel

import importlib

# %%


data = pd.read_pickle(os.path.join('data_generation', 'qc_data_mpc.pkl'))


# # %% 
# # Augment data
qcconf = qcmodel.QuadcopterConfig()
model = qcmodel.get_model(qcconf, with_pos=True)
x_ss, u_ss = qcmodel.get_stable_point(model, p=1)

# lb_aug = ub_aug = np.concatenate((x_ss, u_ss, np.zeros((4,1))), axis=0)

# lb_aug[6] = -np.pi
# ub_aug[6] = np.pi

# lb_aug

# data_aug = np.random.uniform(lb_aug.T, ub_aug.T, size=(1000,20))

# data_aug = pd.DataFrame(data_aug, columns=data.columns)

# data = pd.concat((data, data_aug), axis=0)

del_pos_clip = (data['x_k'][['dx0', 'dx1', 'dx2']] < 0.5).all(axis=1)

data = data[del_pos_clip]

# %%

def train_test_split(data, test_fraction=0.2, **kwargs):
    data_test = data.sample(frac=test_fraction, **kwargs)
    data_train = data.drop(data_test.index)

    return data_train, data_test

# Train test split
data_train, data_test = train_test_split(data, test_fraction=0.8, random_state=42)

# Print information about data
print(f'Train data: {data_train.shape}')
print(f'Test data: {data_test.shape}')


tf_data = tf.data.Dataset.from_tensor_slices((
    data_train['x_k'].to_numpy().astype(np.float32),
    data_train['u_k_prev'].to_numpy().astype(np.float32),
    data_train['u_k'].to_numpy().astype(np.float32),
    np.stack(data_train['du0dx0']['du0dx0']).astype(np.float32),
    np.stack(data_train['du0du0_prev']['dUdU0_prev']).astype(np.float32),
    ))

tf_data = tf_data.shuffle(1000).batch(512)

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
    x_k_in = keras.Input(shape=(data[['x_k']].shape[1]), name='x_k_in')
    u_k_p_in = keras.Input(shape=(data[['u_k_prev']].shape[1]), name='u_k_p_in')
    inputs = keras.layers.concatenate([x_k_in, u_k_p_in], axis=1)
    scale_inputs = keras.layers.Normalization()
    scale_inputs.adapt(data[['x_k', 'u_k_prev']].to_numpy())

    # Out   
    scale_outputs = keras.layers.Normalization(mean=tf.constant(u_ss.reshape(1,-1)), variance=0.1*np.ones((1,4)))
    unscale_outputs = keras.layers.Normalization(invert=True, mean=tf.constant(u_ss.reshape(1,-1)), variance=0.1*np.ones((1,4)))

    layer_in = scale_inputs(inputs)
    #layer_in = inputs

    # Hidden layers
    for k in range(n_layer):
        layer_in = keras.layers.Dense(n_neurons, 
            activation=activation, 
            name=f'hidden_{k}',
            activity_regularizer=keras.regularizers.L2(1e-3),
        )(layer_in)

    # Output layer
    u_k_scaled = keras.layers.Dense(data['u_k'].shape[1], activation='linear')(layer_in)

    u_k_tf = unscale_outputs(u_k_scaled)

    # Create model
    model = keras.Model(inputs=[x_k_in, u_k_p_in], outputs=u_k_tf)

    return model

# %%
model = get_model(data_train, n_layer=4, n_neurons=80, activation='relu')

# %%

@tf.function
def train_step(x1, x2,  y, dydx1, dydx2, model, optimizer):
    with tf.GradientTape(persistent=True) as t1:
        with tf.GradientTape(persistent=True) as t2:
            y_pred = model((x1, x2))
        
        dydx1_pred = t2.batch_jacobian(y_pred, x1)
        dydx2_pred = t2.batch_jacobian(y_pred, x2)

        loss1 = tf.reduce_mean((y-y_pred)**2)
        loss21 = tf.reduce_mean((dydx1-dydx1_pred)**2)
        loss22 = tf.reduce_mean((dydx2-dydx2_pred)**2)
        loss2 = loss21 + loss22

        loss = loss1 + 5*loss2

    gradients = t1.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, loss1, loss2, dydx1_pred

def print_progress(epoch, loss, mse):
    print(f'Epoch: {epoch:4d}, Loss: {loss:.4e}, mse: {mse:.4e}', end='\r')

# %%

optimizer=keras.optimizers.Adam(learning_rate=0.0001)


for k in range(1000):
    for step, (x1, x2, y, dydx1, dydx2) in enumerate(tf_data):
        loss, loss1, loss2, dydx_pred = train_step(x1,x2, y, dydx1, dydx2, model, optimizer)
        print_progress(k, loss, loss1)


# %%
model((x1,x2))

# %%

def plot_parity(data, ax, **kwargs):
    pred = model.predict((data[['x_k']], data[['u_k_prev']]))
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

eval_model.save(os.path.join('models', '02_qc_approx_mpc_model'))
# %%
loaded_model = keras.models.load_model(os.path.join('models', 'qc_approx_mpc_model'))

# %%
test_input = [np.zeros((1,16))]

# %%
loaded_model.layers[-1].invert = True
loaded_model(test_input).numpy()
# %%
eval_model(test_input).numpy()

# %%
# %%
# %%
