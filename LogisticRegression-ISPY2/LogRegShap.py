import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle
import shap

drug = 'Paclitaxel+Pembrolizumab'
gene_df_path = "DrugData/" + drug + "/anova_scores.tsv"
df_path = "DrugData/" + drug + "/" + drug + "_scaled_dataset.tsv"
model_path = "DrugData/" + drug + "/" + "final_model.sav"
importances_path = "DrugData/" + drug + "/feature_importances"

genes = list(pd.read_csv(gene_df_path, delimiter='\t')['GENES'])  # top genes from ETC
df = pd.read_csv(df_path, usecols=[*genes, "TxResponse"], delimiter='\t')
y_df = pd.DataFrame(columns=['TxResponse'])
y_df['TxResponse'] = df['TxResponse']
x_df = df.drop('TxResponse', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=42)

model = pickle.load((open(model_path, 'rb')))


model = model.fit(x_train, y_train)

explainer = shap.Explainer(model, x_train, feature_names=genes)
shap_values = explainer(x_test)
shap.plots.beeswarm(shap_values)#, X_test_array, feature_names=vectorizer.get_feature_names())

print(shap_values.shape)