import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.models import load_model

drug = 'Paclitaxel+Pembrolizumab'
df_path = "DrugData/" + drug + "/" + drug + "_scaled_dataset.tsv"
encoder_input = "DrugData/" + drug + "/" + drug + "_encoder.h5"
df_out = "DrugData/" + drug + "/" + drug + "_encoded_x_df.tsv"

# load data
df = pd.read_csv(df_path, delimiter='\t')

# split data
x_df = df.drop('TxResponse', axis=1)

# gene list
genes = x_df.columns

encoder = load_model(encoder_input)
encoder.compile(optimizer='adam', loss='mse')

x_df_encoded = encoder.predict(x_df)

scaler = MinMaxScaler()

x_df_encoded = scaler.fit_transform(x_df_encoded)

np.savetxt(df_out, x_df, delimiter='\t')
