import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import pickle

# reading in data
drug = "Paclitaxel"
df_path = "../LogisticRegression-ISPY2/DrugData/" + drug + "/" + drug + "_scaled_dataset.tsv"
gene_scores = "../Tensorflow-ISPY2/DrugData/" + drug + "/" + "importances.tsv"
num_genes = 200

genes = pd.read_csv(gene_scores, delimiter='\t')['col_name'].tolist()[:num_genes] + ['TxResponse']
x_matrix = pd.read_csv(df_path, delimiter='\t', usecols=genes)

# getting subtypes and genes
responses = x_matrix['TxResponse'].unique()
genes = genes[:-1]

# setting variables
df_dict = {}

# splitting dataframe by subtype
for response in responses:
    # df = pd.DataFrame()
    df = x_matrix.loc[x_matrix['TxResponse'] == response]
    df = df.drop(['TxResponse'], axis=1)
    filepath = 'DrugData/' + drug + '/' + drug + str(response) + '_x_matrix.csv'
    df.to_csv(filepath, index=False)
    df_dict[response] = df



# calculating fold value and p value for each gene for each subtype
#impact_factor = {}
for response in responses:
    if response == 0:
        other = 1
    else:
        other = 0
    # setting comparison
    other_samples = df_dict[other]
    response_impact_factor = pd.DataFrame(columns=['GENE', 'FACTOR', 'PVALUE'])
    response_gene_samples = df_dict[response]
    if response == other:
        continue
    else:
        for gene in genes:
            total_base = 0.0
            total_response = 0.0
            base_avg = 0.0
            response_avg = 0.0
            factor = 0.0

            gene_p_value = ttest_ind(other_samples[gene], response_gene_samples[gene])
            # gets base subtype gene expression sum
            for sample in list(other_samples[gene]):
                total_base = total_base + float(sample)

            # gets subtype gene expression sum
            for sample in list(response_gene_samples[gene]):
                total_response = total_response + float(sample)

            # compares base subtype and subtype gene expressions
            if total_base != float(0) or total_response != float(0):
                base_avg = total_base / len(other_samples.index)
                response_avg = total_response / len(response_gene_samples.index)
                factor = base_avg / response_avg
                response_impact_factor.loc[len(response_impact_factor)] = [gene, factor, gene_p_value[1]]
        response_impact_factor = response_impact_factor.sort_values(by=['FACTOR'], ascending=False)
        top_20_response_genes = response_impact_factor.head(20)['GENE']
        filepath_genes = 'DrugData/' + drug + '/' + drug + str(response) + '_top_20.csv'
        filepath_full = 'DrugData/' + drug + '/' + drug + str(response) + '_full.csv'
        response_impact_factor.to_csv(filepath_full, index=False)
        top_20_response_genes.to_csv(filepath_genes, index=False)
#impact_factor_sorted = {}
#for key, value in impact_factor.items():
#    impact_factor_sorted[key] = value.sort_values(by=['PVALUE'], ascending=True)

