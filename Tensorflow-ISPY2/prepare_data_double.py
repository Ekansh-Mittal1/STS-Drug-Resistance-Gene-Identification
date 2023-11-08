import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing



x_matrix_file = "../ISPY-Data/mad_X_df.txt"
y_matrix_file = "../ISPY-Data/comp_y_df.csv"
output_matrix_file = "DrugData/Double/Scaled_double_dataset.tsv"
drugs = ['Paclitaxel', 'Paclitaxel+Ganetespib', 'Paclitaxel+MK-2206', 'Paclitaxel+MK-2206+Trastuzumab',
         'Paclitaxel+Neratinib', 'Paclitaxel+Pembrolizumab']

# Read X, Y matrics.
x_df = pd.read_csv(x_matrix_file, delimiter='\t', header=0, index_col=0)
y_df = pd.read_csv(y_matrix_file, header=0, index_col=0)

#
# Process X matrix.
#
# Rename patient ID column in X matrix and select for 7k genes.
x_df.index.name = "Patient"
N = 7000
x_df = x_df.iloc[:, 0:N]

#
# Process Y matrix.
#

#
# Merge X and Y matrices, scale data, and write result to file.
#
joined_df = x_df.join(y_df, how="inner")
joined_df = joined_df.loc[joined_df['Drug'].isin(drugs)]

print(y_df.head())


# Set min-max range to [0,1] in X matrix columns and write out combined matrix.
scaler = MinMaxScaler()
joined_df[x_df.columns] = scaler.fit_transform(joined_df[x_df.columns])
encoder = preprocessing.LabelEncoder()
joined_df['Drug'] = encoder.fit_transform(joined_df['Drug'])
joined_df.to_csv(output_matrix_file, sep="\t", index=False)

a = encoder.inverse_transform([0,1,2,3,4,5])
print(a)