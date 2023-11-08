import pandas as pd
import numpy as np

df = pd.read_csv("../ISPY-Data/y_df.txt", delimiter='\t')

columns = df.columns.tolist()

for i in range(len(columns)):
    columns[i] = columns[i].replace(" + ", "+")
    columns[i] = columns[i].replace(" ", "")

dict = dict(zip(df.columns.tolist(), columns))

df = df.rename(columns=dict)
df['Control'] = None
df['AntiHER2'] = None


df.loc[(df['Paclitaxel'] == 1.0) | (df['Paclitaxel+Trastuzumab'] == 1.0), 'Control'] = 1.0
df.loc[(df['Paclitaxel'] == 0.0) | (df['Paclitaxel+Trastuzumab'] == 0.0), 'Control'] = 0.0

df.loc[(df['Paclitaxel+Neratinib'] == 1.0) | (df['Paclitaxel+Pertuzumab+Trastuzumab'] == 1.0) | (
            1.0 == df['T-DM1+Pertuzumab']), 'AntiHER2'] = 1.0
df.loc[(df['Paclitaxel+Neratinib'] == 0.0) | (df['Paclitaxel+Pertuzumab+Trastuzumab'] == 0.0) | (
            0.0 == df['T-DM1+Pertuzumab']), 'AntiHER2'] = 0.0

df.to_csv('../ISPY-Data/restructured_y_df.csv', sep='\t', index=False)
