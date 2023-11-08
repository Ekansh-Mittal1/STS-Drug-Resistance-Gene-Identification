'''
Prepare data for use in Ludwig.
'''

import pandas as pd
import argparse
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    # Argument parsing.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("x_matrix_file", help="X_matrix file")
    parser.add_argument("y_matrix_file", help="Y_matrix file")
    parser.add_argument("drug", help="Drug to use for filtering")
    parser.add_argument("output_matrix_file", help="Output matrix file")
    args = parser.parse_args()
    
    #Paclitaxel+MK-2206	Paclitaxel+MK-2206+Trastuzumab
    
    x_matrix_file = args.x_matrix_file
    y_matrix_file = args.y_matrix_file
    output_matrix_file = args.output_matrix_file
    drug = args.drug
    '''

    x_matrix_file = "../IspyData/mad_X_df.txt"
    y_matrix_file = "../IspyData/restructured_y_df.tsv"
    drug = "Paclitaxel+Neratinib"
    output_matrix_file = "../LogisticRegression-ISPY2/DrugData/" + drug + "/" + drug + "_scaled_dataset.tsv"

    # Read X, Y matrics.
    x_df = pd.read_csv(x_matrix_file, delimiter="\t", header=0, index_col=0)
    y_df = pd.read_csv(y_matrix_file, delimiter="\t", header=0, index_col=0)

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
    
    #
    # Merge X and Y matrices, scale data, and write result to file.
    #
    joined_df = x_df.join(y_df, how="inner")

    # Set min-max range to [0,1] in X matrix columns and write out combined matrix.
    scaler = MinMaxScaler()
    joined_df[x_df.columns] = scaler.fit_transform(joined_df[x_df.columns])
    joined_df.to_csv(output_matrix_file, sep="\t", index=False)
