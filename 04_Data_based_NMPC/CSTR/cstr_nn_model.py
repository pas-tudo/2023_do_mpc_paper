import do_mpc
from casadi import *
import numpy as np
import onnx



def get_nn_model(onnx: onnx.ModelProto):
    """
    Get a surrogate model of the CSTR system. The data-based model is a neural network and must be provided as an ONNX model.

    Parameters
    ----------
    onnx : onnx.ModelProto
        ONNX model of the neural network.
    
    Returns
    -------
    model : do_mpc.model.Model
        do-mpc model of the CSTR system with the neural network as surrogate model.
    """
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type, 'SX')


    # States struct (optimization variables):
    C_a = model.set_variable(var_type='_x', var_name='C_a', shape=(1,1))
    C_b = model.set_variable(var_type='_x', var_name='C_b', shape=(1,1))
    T_R = model.set_variable(var_type='_x', var_name='T_R', shape=(1,1))
    T_K = model.set_variable(var_type='_x', var_name='T_K', shape=(1,1))
    # Concatenate states:
    x = vertcat(C_a, C_b, T_R, T_K)

    # TVP
    C_b_set = model.set_variable(var_type='_tvp', var_name='C_b_set', shape=(1,1))

    # Introduce parameters (not used) to comply with physical model
    alpha = model.set_variable(var_type='_p', var_name='alpha', shape=(1,1))
    beta = model.set_variable(var_type='_p', var_name='beta', shape=(1,1))

    # Input struct (optimization variables):
    F = model.set_variable(var_type='_u', var_name='F')
    Q_dot = model.set_variable(var_type='_u', var_name='Q_dot')
    # Concatenate inputs:
    u = vertcat(F, Q_dot)

    # Use ONNX model with converter
    casadi_converter = do_mpc.sysid.ONNXConversion(onnx)
    casadi_converter.convert(x_k = x.T, u_k = u.T)
    x_next = casadi_converter['x_next']
    

    # Differential equations
    model.set_rhs('C_a', x_next[0])
    model.set_rhs('C_b', x_next[1])
    model.set_rhs('T_R', x_next[2]) 
    model.set_rhs('T_K', x_next[3])

    # Build the model
    model.setup()
    return model