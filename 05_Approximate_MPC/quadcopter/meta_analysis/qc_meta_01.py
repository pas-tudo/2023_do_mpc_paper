# %%
import sys 
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from casadi import *
import do_mpc
import json


sys.path.append(os.path.join('..', '.'))
sys.path.append(os.path.join('..','data_generation', '.'))

from qc_approx_helper import get_model
from qc_data_prep import get_data_for_approx_mpc
import qc_train_approx_mpc

# %%

data_dir = os.path.join('..', 'data_generation', 'closed_loop_mpc_02')

plan = do_mpc.tools.load_pickle(os.path.join(data_dir, 'sampling_plan_mpc.pkl'))

dh = do_mpc.sampling.DataHandler(plan)
dh.data_dir = os.path.join(data_dir, '')

assert len(dh[:]) == 50, 'Assuming the number of trajectories is 50.'

# %%

pd.DataFrame(dh[:])

# %%

gamma_sobolov_list = [0, 100]
number_of_trajectories_list = [5, 10, 20, 40]

for number_of_trajectories in number_of_trajectories_list:
    # Load a number of trajectories
    data_train = get_data_for_approx_mpc(dh[:number_of_trajectories])
    data_test = get_data_for_approx_mpc(dh[-10:])
    # Preprocess data
    data_train_tf = qc_train_approx_mpc.create_tf_dataset(data_train, batch_size=100_000)
    data_test_tf = qc_train_approx_mpc.create_tf_dataset(data_test, batch_size=100_000)

    for gamma_sobolov in gamma_sobolov_list:
        print('----------------------------------------------')
        print('Training model with following parameters:')
        print(f'gamma_sobolov:          {gamma_sobolov}')
        print(f'Number of trajectories: {number_of_trajectories}')

        # 
        model = get_model(data_train, n_layer=1, n_neurons=80, activation='tanh')

        # %%

        optimizer=keras.optimizers.Adam(learning_rate=0.001)

        sobolov_trainer = qc_train_approx_mpc.CustomTrainer(model)
        sobolov_trainer.setup_training(optimizer, gamma_sobolov=gamma_sobolov)

        # %%

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-8,
            patience=1000,
            verbose=1,
            mode="auto",
            restore_best_weights=True,
        )
        epoch, train_mse, val_mse = sobolov_trainer.fit(data_train_tf, val=data_test_tf, epochs=5000, callbacks=[early_stopping])

        meta = {
            'epoch': epoch,
            'train_mse': float(train_mse.numpy()),
            'val_mse': float(val_mse.numpy()),
            'gamma_sobolov': gamma_sobolov,
            'number_of_trajectories': number_of_trajectories
        }

        model_name = f'qc_meta_traj_{str(number_of_trajectories)}_gamma_{gamma_sobolov}'

        model.save(os.path.join('models_meta_02', model_name))

        with open(os.path.join('models_meta_02', model_name, 'custom_meta.json'), 'w') as f:
            json.dump(meta, f)
