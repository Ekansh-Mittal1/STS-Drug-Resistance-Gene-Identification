import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import yaml
import pickle
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.models import load_model

drug = 'Paclitaxel+Pembrolizumab'
df_path = "DrugData/" + drug + "/" + drug + "_scaled_dataset.tsv"
encoder_input = "DrugData/" + drug + "/" + drug + "_encoder.h5"
param_input_path = "DrugData/" + drug + "/" + "Encoder_Model_hyperparamters.yaml"
model_save_path = "DrugData/" + drug + "/" + "encoder_final_model.sav"

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

params = yaml.safe_load(open(param_input_path))

model = LogisticRegression(C=params['C'], max_iter=params['max_iter'], penalty=params['penalty'],
                           solver=params['solver'])
model.fit(x_train, y_train)

y_hat = model.predict(x_test)

acc = accuracy_score(y_test, y_hat)
print(acc)