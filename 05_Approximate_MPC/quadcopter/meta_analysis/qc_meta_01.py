# %%
import sys 
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from casadi import *


sys.path.append(os.path.join('..', '.'))

from qc_approx_helper import get_model
import qc_train_approx_mpc
# %%

gamma_sobolov_list = [0, 100]
test_fraction_list = [0.2, 0.4, 0.6, 0.8]



# Load data
data = pd.read_pickle(os.path.join('..', 'data_generation', 'qc_data_mpc.pkl'))

for test_fraction in test_fraction_list:
    # Train test split
    data_train, data_test = qc_train_approx_mpc.train_test_split(data, test_fraction=test_fraction, random_state=42)
    # Preprocess data
    data_train_tf = qc_train_approx_mpc.create_tf_dataset(data_train, batch_size=10_000)

    for gamma_sobolov in gamma_sobolov_list:
        print('----------------------------------------------')
        print('Training model with following parameters:')
        print(f'gamma_sobolov: {gamma_sobolov}')
        print(f'test_fraction: {test_fraction}')

        # 
        model = get_model(data_train, n_layer=1, n_neurons=80, activation='tanh')
        model.summary()


        # %%

        optimizer=keras.optimizers.Adam(learning_rate=0.001)

        sobolov_trainer = qc_train_approx_mpc.CustomTrainer(model)
        sobolov_trainer.setup_training(optimizer, gamma_sobolov=gamma_sobolov)

        # %%

        sobolov_trainer.fit(data_train_tf, epochs=8000)

        model_name = f'qc_meta_testfr_{str(test_fraction).replace(".", "_")}_gamma_{gamma_sobolov}'

        model.save(os.path.join('models_meta', model_name))
