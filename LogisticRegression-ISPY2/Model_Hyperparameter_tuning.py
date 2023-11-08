import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import yaml
from sklearn.metrics import accuracy_score

drug = 'Paclitaxel+Pembrolizumab'
gene_df_path = "DrugData/" + drug + "/anova_scores.tsv"
df_path = "DrugData/" + drug + "/" + drug + "_scaled_dataset.tsv"
param_output_path = "DrugData/" + drug + "/" + "Model_hyperparamters.yaml"

# import data
genes = list(pd.read_csv(gene_df_path, delimiter="\t")['GENES'])[:200] # top genes from ETC
df = pd.read_csv(df_path, usecols=[*genes, "TxResponse"], delimiter='\t')
y_df = pd.DataFrame(columns=['TxResponse'])
y_df['TxResponse'] = df['TxResponse']
x_df = df.drop('TxResponse', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=42)

model = LogisticRegression()

param_grid = [
    {'penalty': ['l1', 'l2', 'elasticnet'],
     'C': np.logspace(-4, 4, 20),
     'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
     'max_iter': [5000, 7500, 10000],
    }
]

clf = GridSearchCV(model, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
best_clf = clf.fit(x_train, y_train)
print('score: ', clf.best_score_)
print('parameters: ', clf.best_params_)
tuned_model = LogisticRegression(penalty=clf.best_params_['penalty'], C=clf.best_params_['C'], solver=clf.best_params_['solver'], max_iter=clf.best_params_['max_iter'])
tuned_model.fit(x_train, y_train)
print("score",tuned_model.score(x_test,y_test))

params = clf.best_params_
params['C'] = float(clf.best_params_['C'])
print(params)

with open(param_output_path, "w") as output_file:
    yaml.dump(params, output_file)