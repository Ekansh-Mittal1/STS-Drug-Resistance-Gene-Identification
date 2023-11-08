import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

drug = 'Paclitaxel+Pembrolizumab'
df_path = "DrugData/" + drug + "/" + drug + "_scaled_dataset.tsv"
output_path = "DrugData/" + drug + "/" + drug + "_normalized_genes.tsv"

# load data
df = pd.read_csv(df_path, delimiter='\t')

# split data
y_df = df[['TxResponse']].copy()
x_df = df.drop('TxResponse', axis=1)

# gene list
genes = x_df.columns

# fit model
etc_importance_sums = dict.fromkeys(list(genes), 0.0)
etc_importance_list = []
etc_importance_df = pd.DataFrame()

for i in range(100):
    etc = ExtraTreesClassifier(n_estimators=5, criterion='entropy')
    etc.fit(x_df, y_df.values.ravel())

    # normalize feature importances
    feature_importances = etc.feature_importances_
    feature_importance_normalized = np.std([tree.feature_importances_ for tree in
                                            etc.estimators_],
                                           axis=0)

    etc_importance_df[i] = feature_importance_normalized

    #etc_importance_list.append(list(feature_importance_normalized))
    #for j in range(len(list(feature_importance_normalized))):
    #   etc_importance_sums[genes[i]] = etc_importance_sums[genes[i]] + float(feature_importances[i])

    #   print(feature_importances[i])

print(etc_importance_df)
etc_importance_df.to_csv("test.csv")
# send to pandas df
'''
feature_importances_df = pd.DataFrame({'GENES': genes, 'IMPORTANCE': list(feature_importances)})
feature_importances_df = feature_importances_df.sort_values(by='IMPORTANCE', ascending=False)
top_genes = feature_importances_df[feature_importances_df['IMPORTANCE'] > 0]
top_genes.to_csv('top_genes_' + drug + '.csv', index=False)
'''

print(feature_importance_normalized)

normalized_df = pd.DataFrame({'GENES': genes, 'IMPORTANCE': list(feature_importance_normalized)})
normalized_df = normalized_df.sort_values(by='IMPORTANCE', ascending=False)
top_genes_normalized = normalized_df[normalized_df['IMPORTANCE'] > 0]
top_genes_normalized.to_csv(output_path, index=False, sep="\t")

# visualize
plt.bar(top_genes_normalized['GENES'][:20], top_genes_normalized['IMPORTANCE'][:20])
plt.xlabel('Feature Labels')
plt.ylabel('Feature Importances')
plt.title('Top 20 Feature Importances: ExtraTreesClassifier')
plt.show()

sns.heatmap(x_df[list(normalized_df['GENES'])[:20]].corr(), cmap='Blues', annot=True)
plt.show()
