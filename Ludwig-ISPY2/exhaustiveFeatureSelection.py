import pandas as pd
import ludwig
from ludwig.api import LudwigModel
from sklearn.preprocessing import MinMaxScaler
import yaml

drug = 'Paclitaxel'
print('loading data')
x_df = pd.read_csv("../ISPY-Data/mad_X_df.txt", delimiter='\t', header=0, index_col=0)
y_df = pd.read_csv("../ISPY-Data/y_df.txt", delimiter='\t', header=0, index_col=0)
print('data loaded')
print('preparing data')
#
# Process X matrix.
#

# Rename patient ID column in X matrix and select for 7k genes.
x_df.index.name = "Patient"

# TODO: (1) Make this a parameter and (2) note that this only works if using mad_X_df.txt, not X df.
N = 7000
x_df = x_df.iloc[:, 0:N]

#
# Process Y matrix.
#

# Fill in empty values with -1, rename patient ID column, and sort by patient ID.
y_df.fillna(-1, inplace=True)
y_df.index.name = "Patient"

# Filter to get patient response data for the given drug and rename column.
y_df = y_df[y_df[drug] >= 0][drug]
y_df.rename("TxResponse", inplace=True)

joined_df = x_df.join(y_df, how="inner")
print('data prepped')

print('scaling data')
# Set min-max range to [0,1] in X matrix columns and write out combined matrix.
scaler = MinMaxScaler()
joined_df[x_df.columns] = scaler.fit_transform(joined_df[x_df.columns])
print('data scaled')

print('creating config')
# Read features from input.
feature_names = joined_df.columns.tolist()

# Define inputs.
inputs = [{"name": c, "type": "numerical"} for c in feature_names[:-1]]
# Get outputs from base config
base_config = yaml.safe_load(open("base_config.yaml"))
outputs = base_config['output_features']
training = base_config['training']

#write config
config = {
    "input_features": inputs,
    "output_features": outputs,
    "training": training
}
print('config made')

print('training model')
ludwig_model = LudwigModel(config, logging_level=40)
train_stats, _, _ = ludwig_model.train(dataset=joined_df)
print('model trained')