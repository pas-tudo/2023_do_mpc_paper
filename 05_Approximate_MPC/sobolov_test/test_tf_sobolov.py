# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
# %%

import do_mpc
import casadi
print(do_mpc.__version__)
# %%


def get_model(n_in, n_out):
    input = keras.Input(shape=(n_in,))
    mid = keras.layers.Dense(10, activation='tanh')(input)
    mid = keras.layers.Dense(10, activation='tanh')(mid)
    output = keras.layers.Dense(n_out, activation='linear')(mid)

    model = keras.Model(inputs=input, outputs=output)

    return model
# %%

@tf.function
def train_step(x,y, dydx, model, optimizer):
    with tf.GradientTape() as t1:
        with tf.GradientTape() as t2:
            y_pred = model(x)
        
        dydx_pred = t2.gradient(y_pred, x)

        loss = tf.keras.losses.mse(y, y_pred)
        loss += tf.keras.losses.mse(dydx, dydx_pred) 
        loss = tf.reduce_mean(loss)
            
    gradients = t1.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss



# %%
x = np.random.uniform(0, 6*np.pi, size=(10,1)).astype(np.float32)
y = np.sin(x)
dydx = np.cos(x)

model = get_model(1,1)

# %%

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

x = tf.Variable(x)

# %%
for k in range(5000):
    loss = train_step(x,y, dydx, model, optimizer)

# %%
loss
# %%

y_pred = model(x)

# %%
plt.plot(x.numpy(), y, 'o', label='train')

x_test = np.linspace(0, 6*np.pi, 100).astype(np.float32)
y_pred = model(x_test).numpy()
plt.plot(x_test, y_pred, label='pred')
plt.legend()

# %% 
import sys
import os
