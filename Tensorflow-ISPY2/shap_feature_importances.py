import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import shap
import numpy as np
from matplotlib import pyplot as plt


drug = 'Paclitaxel'
x_df_path = "DrugData/" + drug + "/" + "rebalanced_x_df.tsv"
y_df_path = "DrugData/" + drug + "/" + "rebalanced_y_df.tsv"
gene_scores = "../LogisticRegression-ISPY2/DrugData/" + drug + "/" + "chi2_scores.tsv"
param_input_path = "DrugData/" + drug + "/" + "params.yaml"
model_save_path = "DrugData/" + drug + "/optimized_model.h5"
num_genes = 400

genes = pd.read_csv(gene_scores, delimiter='\t')['GENES'].tolist()[:num_genes]
x_df = pd.read_csv(x_df_path, delimiter='\t', usecols=genes)
y_df = pd.read_csv(y_df_path, delimiter='\t', index_col=0)

x_train, x_rem, y_train, y_rem = train_test_split(x_df, y_df, train_size=0.8, random_state=42)

x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size=0.5, random_state=42)

model = load_model(model_save_path)

model.evaluate(x_train, y_train, verbose=2)
model.evaluate(x_test, y_test, verbose=2)
model.evaluate(x_valid, y_valid, verbose=2)

x_train_np = x_train.to_numpy()
x_test_np = x_test.to_numpy()
x_valid_np = x_valid.to_numpy()

explainer = shap.GradientExplainer(model, x_train_np, features=genes)
shap_values = explainer.shap_values(x_valid_np)
shap.summary_plot(shap_values[0], plot_type='bar', feature_names=genes, max_display=20)

feature_names = x_train.columns

rf_resultX = pd.DataFrame(shap_values[0], columns=feature_names)

print(rf_resultX)

vals = np.abs(rf_resultX.values).mean(0)

shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                               columns=['col_name', 'feature_importance_vals'])
shap_importance.sort_values(by=['feature_importance_vals'],
                            ascending=False, inplace=True)
print(shap_importance.head())

shap_importance.to_csv('DrugData/' + drug + '/importances.tsv', sep='\t')
