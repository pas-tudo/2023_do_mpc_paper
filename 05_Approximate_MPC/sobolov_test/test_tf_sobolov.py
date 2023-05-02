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
np.random.seed(99)
x = tf.Variable(np.random.uniform(size=(10,1)).astype(np.float32))
y = np.ones(x.shape).astype(np.float32)

with tf.GradientTape(persistent=True) as t1:
    with tf.GradientTape(persistent=True) as t2:
        y_pred = x*tf.math.sin(x)+tf.math.cos(x**3)
    
    dydx = t2.gradient(y_pred, x)

    loss = tf.reduce_mean((y_pred-y)**2)
    loss += tf.reduce_mean((dydx-y)**2)

dloss_dx = t1.gradient(loss, x)

print(dloss_dx)


# %%

def get_model(n_in, n_out):
    tf.random.set_seed(99)
    tf.keras.utils.set_random_seed(99)
    input = keras.Input(shape=(n_in,))
    mid = keras.layers.Dense(10, activation='tanh')(input)
    mid = keras.layers.Dense(10, activation='tanh')(mid)
    output = keras.layers.Dense(n_out, activation='linear')(mid)

    model = keras.Model(inputs=input, outputs=output)

    return model
# %%
@tf.function
def train_step(x,y, dydx, model, optimizer):
    with tf.GradientTape(persistent=True) as t1:
        with tf.GradientTape(persistent=True) as t2:
            y_pred = model(x)
        
        dydx_pred = t2.batch_jacobian(y_pred, x)

        # loss1 = tf.reduce_mean((y_pred-y)**2)
        # loss2 = tf.reduce_mean((dydx-dydx_pred)**2)

        loss1 = tf.reduce_mean(tf.keras.losses.mse(y_pred,y))
        loss2 = tf.reduce_mean(tf.keras.losses.mse(dydx,dydx_pred))
        loss = loss1+loss2

    gradients = t1.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, loss1, loss2



# %%
np.random.seed(99)
tf.random.set_seed(99)
x = np.random.uniform(0, 6*np.pi, size=(10,1)).astype(np.float32)
y = np.sin(x)
dydx = np.cos(x).reshape(-1,1,1)

model = get_model(1,1)


# %%

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

x = tf.Variable(x)


for k in range(1000):
    loss, loss1, loss2= train_step(x,y, dydx, model, optimizer)


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
