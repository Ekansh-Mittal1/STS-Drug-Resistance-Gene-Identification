import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
import numpy as np
import matplotlib.pyplot as plt
from numpy import array

drug = "Paclitaxel"
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
chi2_selector = SelectKBest(chi2, k=20).fit(x_df, y_df)
filter_chi2 = chi2_selector.get_support()
features = array(x_df.columns)
chi2_scores = pd.DataFrame(list(zip(x_df.columns, chi2_selector.scores_)), columns=['GENES', 'Score'])
chi2_scores = chi2_scores.sort_values(by='Score', ascending=False)
chi2_scores.to_csv("DrugData/"+drug+"/chi2_scores.tsv", sep='\t')

anova_selector = SelectKBest(f_classif, k=20).fit(x_df, y_df)
filter_anova = anova_selector.get_support()
anova_scores = pd.DataFrame(list(zip(x_df.columns, anova_selector.scores_)), columns=['GENES', 'Score'])
anova_scores = anova_scores.sort_values(by='Score', ascending=False)
anova_scores.to_csv("DrugData/"+drug+"/anova_scores.tsv", sep='\t')


# send to pandas df

# visualize
plt.bar(chi2_scores['GENES'][:20], chi2_scores['Score'][:20])
print(chi2_scores['Score'][:20])
plt.xlabel('Gene')
plt.ylabel('Score')
plt.title('Top 20 Feature Importances: Chi Squared')
plt.savefig("DrugData/"+drug+"/FeatureSelectionChi2.png")
plt.show()

plt.bar(anova_scores['GENES'][:20], anova_scores['Score'][:20])
print(anova_scores['Score'][:20])
plt.xlabel('Gene')
plt.ylabel('Score')
plt.title('Top 20 Feature Importances: Anova')
plt.savefig("DrugData/"+drug+"/FeatureSelectionAnova.png")
plt.show()