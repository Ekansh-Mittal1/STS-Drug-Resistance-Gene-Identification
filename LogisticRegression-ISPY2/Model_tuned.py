import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import yaml
import pickle
import shap

drug = 'Paclitaxel+Pembrolizumab'
gene_df_path = "DrugData/" + drug + "/anova_scores.tsv"
df_path = "DrugData/" + drug + "/" + drug + "_scaled_dataset.tsv"
param_input_path = "DrugData/" + drug + "/" + "Model_hyperparamters.yaml"
model_save_path = "DrugData/" + drug + "/" + "final_model.sav"

# import data
genes = list(pd.read_csv(gene_df_path, delimiter='\t')['GENES'])[:200]  # top genes from ETC
df = pd.read_csv(df_path, usecols=[*genes, "TxResponse"], delimiter='\t')
y_df = pd.DataFrame(columns=['TxResponse'])
y_df['TxResponse'] = df['TxResponse']
x_df = df.drop('TxResponse', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=42)

params = yaml.safe_load(open(param_input_path))

model = LogisticRegression(C=params['C'], max_iter=params['max_iter'], penalty=params['penalty'],
                           solver=params['solver'])
model.fit(x_train, y_train)


explainer = shap.Explainer(model, x_train, feature_names=genes)
shap_values = explainer(x_train)

shap.plots.beeswarm(shap_values)

pickle.dump(model, open(model_save_path, 'wb'))
