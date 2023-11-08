import pandas as pd
import tensorflow as tf
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import math
import yaml
import matplotlib.pyplot as plt

tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

df_path = "DrugData/Double/Scaled_double_dataset.tsv"
gene_scores = "DrugData/Double/Double_gene_scores.tsv"
param_output_path = "DrugData/Double/params.yaml"
model_save_path = "DrugData/Double/optimized_model.h5"
num_genes = 400

genes = pd.read_csv(gene_scores, delimiter='\t')['GENES'].tolist()[:num_genes]

print(genes)
x_df = pd.read_csv(df_path, delimiter='\t', usecols=genes)

y_df = pd.read_csv(df_path, delimiter='\t', usecols=['Drug', 'Response'])

x_train, x_rem, y_train, y_rem = train_test_split(x_df, y_df, train_size=0.8, random_state=42)

x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size=0.5, random_state=42)


def train_eval_model(num_layers,
                     num_epochs,
                     batch_size,
                     lr,
                     layer_activation,
                     output_activation,
                     dropout,
                     x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid,
                     num_inputs=400,
                     save=False, save_path=model_save_path, plot=False):
    inputs = tf.keras.Input(shape=(num_inputs))
    x = layers.Dense(num_inputs / 2, activation=layer_activation)(inputs)
    x = layers.Dropout(dropout)(x)

    for i in range(1, num_layers):
        nodes = int(math.ceil(num_inputs / 2 ** i))
        x = layers.Dense(nodes, activation=layer_activation)(x)
        x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(1, activation=output_activation)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss='mae', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1,
                        validation_data=(x_valid, y_valid))
    val_loss, _ = model.evaluate(x_valid, y_valid, verbose=2)

    if plot:
        plt.plot(np.log10(history.history['loss']))
        plt.plot(np.log10(history.history['val_loss']))
        plt.title('model loss')
        plt.ylabel('loss (log_10)')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    if save:
        model.save(save_path)
        return model
    else:
        return 1 - val_loss


def optimize_nn(x_train, y_train, x_valid, y_valid):
    layer_activs = ['relu', 'sigmoid', 'tanh']
    output_activs = ['sigmoid', 'softmax']

    def nn_wrapper(num_layers, num_epochs, batch_size_exp, lr, layer_activNum, output_activNum, dropout):
        num_layers = int(num_layers)
        num_epochs = int(num_epochs)

        batch_size = int(2 ** batch_size_exp)

        layer_activation = layer_activs[int(layer_activNum)]
        output_activation = output_activs[int(output_activNum)]

        return train_eval_model(num_layers=num_layers,
                                num_epochs=num_epochs,
                                batch_size=batch_size,
                                lr=lr,
                                layer_activation=layer_activation,
                                output_activation=output_activation,
                                dropout=dropout,
                                x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid
                                )

    optimizer = BayesianOptimization(
        f=nn_wrapper,
        pbounds={"num_layers": (1, 6), "num_epochs": (200, 500), "batch_size_exp": (4, 9), "lr": (1e-4, 1e-2),
                 "layer_activNum": (0, 2), "output_activNum": (0, 1), "dropout": (0, 0.5)},
        random_state=1234,
        verbose=2
    )

    logger = JSONLogger(path="DrugData/Paclitaxel+Pembrolizumab/logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(n_iter=10, init_points=10)
    print("Final result:", optimizer.max)

    params = {}
    params['batch_size'] = int(2 ** optimizer.max['params']['batch_size_exp'])
    params['layer_activation'] = layer_activs[int(optimizer.max['params']['layer_activNum'])]
    params['lr'] = float(optimizer.max['params']['lr'])
    params['num_epochs'] = int(optimizer.max['params']['num_epochs'])
    params['num_layers'] = int(optimizer.max['params']['num_layers'])
    params['output_activation'] = output_activs[int(optimizer.max['params']['output_activNum'])]
    params['dropout'] = float(optimizer.max['params']['dropout'])
    with open(param_output_path, "w") as output_file:
        yaml.dump(params, output_file)

    model = train_eval_model(num_layers=params['num_layers'],
                             num_epochs=params['num_epochs'],
                             batch_size=params['batch_size'],
                             lr=params['lr'],
                             layer_activation=params['layer_activation'],
                             output_activation=params['output_activation'],
                             dropout=params['dropout'],
                             save=True, plot=True)

    model.evaluate(x_train, y_train, verbose=2)
    model.evaluate(x_test, y_test, verbose=2)
    model.evaluate(x_valid, y_valid, verbose=2)


optimize_nn(x_train, y_train, x_valid, y_valid)
