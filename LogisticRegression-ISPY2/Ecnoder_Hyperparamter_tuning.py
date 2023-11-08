import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
import yaml
import numpy as np

import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.models import load_model



drug = 'Paclitaxel+Pembrolizumab'
df_path = "DrugData/" + drug + "/" + drug + "_scaled_dataset.tsv"
encoder_input = "DrugData/" + drug + "/" + drug + "_encoder.h5"
param_output_path = "DrugData/" + drug + "/" + "Encoder_Model_hyperparamters.yaml"

# load data
df = pd.read_csv(df_path, delimiter='\t')

# split data
y_df = df[['TxResponse']].copy()
x_df = df.drop('TxResponse', axis=1)

# gene list
genes = x_df.columns

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=42)

encoder = load_model(encoder_input)
encoder.compile(optimizer='adam', loss='mse')

x_train = encoder.predict(x_train)
x_test = encoder.predict(x_test)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

model = LogisticRegression()

param_grid = [
    {'penalty': ['l1', 'l2', 'elasticnet'],
     'C': np.logspace(-4, 4, 20),
     'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
     'max_iter': [5000, 7500, 10000],
    }
]

clf = GridSearchCV(model, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
clf.fit(x_train, y_train)
print('score: ', clf.best_score_)
print('parameters: ', clf.best_params_)
tuned_model = LogisticRegression(penalty=clf.best_params_['penalty'], C=clf.best_params_['C'], solver=clf.best_params_['solver'], max_iter=clf.best_params_['max_iter'])
tuned_model.fit(x_train, y_train)
print("score",tuned_model.score(x_test,y_test))

params = clf.best_params_
params['C'] = float(clf.best_params_['C'])
print(params)

with open(param_output_path, "w") as output_file:
    yaml.dump(params, output_file)
