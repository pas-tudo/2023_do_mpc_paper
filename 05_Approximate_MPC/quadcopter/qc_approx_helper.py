import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Optional


def get_model(
        data: pd.DataFrame, 
        n_layer: int = 4, 
        n_neurons: int = 32, 
        activation='tanh',
        ) -> keras.Model:
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
    model : keras.Model
        Model used for training (returns scaled outputs)
    """


    # Model input
    x_k_in = keras.Input(shape=(data[['x_k']].shape[1]), name='x_k_in')
    u_k_p_in = keras.Input(shape=(data[['u_k_prev']].shape[1]), name='u_k_p_in')
    inputs = keras.layers.concatenate([x_k_in, u_k_p_in], axis=1)
    layer_in = inputs

    # Hidden layers
    for k in range(n_layer):
        layer_in = keras.layers.Dense(n_neurons, 
            activation=activation, 
            name=f'hidden_{k}',
            activity_regularizer=keras.regularizers.L2(1e-3),
        )(layer_in)

    # Output layer
    u_k_tf = keras.layers.Dense(data['u_k'].shape[1], activation='linear')(layer_in)

    # Create model
    model = keras.Model(inputs=[x_k_in, u_k_p_in], outputs=u_k_tf)

    return model



class ApproxMPC:
    """
    Approximate MPC controller.

    Requires a Keras NN that maps:

    .. math::

        (x_k, u_{k-1}) \\rightarrow u_k

    with ``x_k.shape = (n_x, 1)``, ``u_k.shape = (n_u, 1)``. 
    
    """
    def __init__(self, model: keras.Model, n_x: int, n_u: int, u0: np.ndarray):

        assert isinstance(model, keras.Model), 'model must be keras.Model'
        assert isinstance(n_x, int), 'n_x must be int'
        assert isinstance(n_u, int), 'n_u must be int'
        assert isinstance(u0, np.ndarray), 'u0 must be numpy array'
        assert u0.shape == (n_u, 1), 'u0 must be of shape (n_u, 1)'


        self.keras_model = model

        self.n_x = n_x
        self.n_u = n_u

        self.u0 = u0

    def make_step(self, x0):
        """
        Pass current state to approximate MPC controller and return control input.
        """

        assert isinstance(x0, np.ndarray), 'x0 must be numpy array'
        assert x0.shape == (self.n_x, 1), 'x0 must be of shape (n_x, 1)'

        dp_max = np.array([.3,.3,.1]).reshape(-1,1)
        x0[:3] = np.clip(x0[:3], -dp_max, dp_max)

        # Prepare NN input
        u_prev = self.u0
        nn_in = (
            x0.reshape(1,-1),
            u_prev.reshape(1,-1),
        )

        # Evaluate NN
        u0 = self.keras_model(nn_in).numpy().reshape(-1,1)

        # Store u0 for next iteration
        self.u0 = u0

        return u0