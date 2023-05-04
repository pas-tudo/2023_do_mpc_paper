# %% [markdown]
# # Train approximate MPC controller for quadcopter
# Load packages

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


from qc_approx_helper import get_model

# %% [markdown]
# ## Functions to preprocess data

# %%

def train_test_split(data, test_fraction=0.2, **kwargs):
    data_test = data.sample(frac=test_fraction, **kwargs)
    data_train = data.drop(data_test.index)

    # Print information about data
    print(f'Train data: {data_train.shape}')
    print(f'Test data: {data_test.shape}')

    return data_train, data_test

def create_tf_dataset(data_train, batch_size=512):
    data_train_tf = tf.data.Dataset.from_tensor_slices((
        data_train['x_k'].to_numpy().astype(np.float32),
        data_train['u_k_prev'].to_numpy().astype(np.float32),
        data_train['u_k'].to_numpy().astype(np.float32),
        np.stack(data_train['du0dx0']['du0dx0']).astype(np.float32),
        np.stack(data_train['du0du0_prev']['dUdU0_prev']).astype(np.float32),
        ))

    data_train_tf = data_train_tf.shuffle(1000).batch(batch_size)

    return data_train_tf



# %% [markdown]
# ## Functions and classes to train model

# %%

def print_progress(epoch: int, **kwargs):
    print_str = f'Epoch {epoch:4d}: '

    for key, value in kwargs.items():
        print_str += f'{key} : {value:.3e}, '

    print(print_str, end='\r')

class CustomTrainer:
    def __init__(self, model):
        self.model = model

        self.is_traing_setup = False

    def setup_training(self, optimizer, gamma_sobolov = 100):
        self.optimizer = optimizer

        self.gamma_sobolov = gamma_sobolov

        self.is_traing_setup = True

    
    @tf.function
    def train_step_sobolov(self, x1, x2, y, dydx1, dydx2):
        with tf.GradientTape(persistent=True) as t1:
            with tf.GradientTape(persistent=True) as t2:
                y_pred = self.model((x1, x2))
            
            dydx1_pred = t2.batch_jacobian(y_pred, x1)
            dydx2_pred = t2.batch_jacobian(y_pred, x2)

            loss1 = tf.reduce_mean((y-y_pred)**2)
            loss21 = tf.reduce_mean((dydx1-dydx1_pred)**2)
            loss22 = tf.reduce_mean((dydx2-dydx2_pred)**2)
            loss2 = loss21 + loss22

            loss = loss1 + self.gamma_sobolov*loss2

        gradients = t1.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss, loss1, loss2
    
    @tf.function
    def train_step(self, x1, x2, y):
        with tf.GradientTape(persistent=True) as t:
            y_pred = self.model((x1,x2))
            loss = tf.reduce_mean((y-y_pred)**2)

        gradients = t.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def fit(self, data_train, val = None, epochs = 10):
        if not self.is_traing_setup:
            raise Exception('Training not setup. Run setup_training() first.')

        for k in range(epochs):
            for step, (x1, x2, y, dydx1, dydx2) in enumerate(data_train):
                x1 = tf.Variable(x1)
                x2 = tf.Variable(x2)
                if self.gamma_sobolov > 0:
                    loss, loss1, loss2  = self.train_step_sobolov(x1,x2, y, dydx1, dydx2)
                    print_progress(k, loss = loss, mse = loss1, sobolov_mse = loss2)        
                else:
                    loss = self.train_step(x1,x2, y)
                    print_progress(k, mse = loss)        

        
# %%

if __name__ == '__main__':
    # Load data
    data = pd.read_pickle(os.path.join('data_generation', 'qc_data_mpc.pkl'))
    # Train test split
    data_train, data_test = train_test_split(data, test_fraction=0.2, random_state=42)
    # Preprocess data
    data_train_tf = create_tf_dataset(data_train, batch_size=2000)

    # 
    model = get_model(data_train, n_layer=1, n_neurons=80, activation='tanh')
    model.summary()


    # %%

    optimizer=keras.optimizers.Adam(learning_rate=0.001)

    sobolov_trainer = CustomTrainer(model)
    sobolov_trainer.setup_training(optimizer, gamma_sobolov=100)

    # %%

    sobolov_trainer.fit(data_train_tf, epochs=5000)


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
    model_name = '05_qc_approx_mpc_model'

    model.save(os.path.join('models', model_name))
    # %%
    loaded_model = keras.models.load_model(os.path.join('models', model_name))

# %%
