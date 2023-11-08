import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score, precision_score, make_scorer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
import pickle

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
y_hat = model.predict(x_test)


accuracy = accuracy_score(y_test, y_hat)
r2 = r2_score(y_test, y_hat)
precision = precision_score(y_test, y_hat, pos_label=1.0)

print(accuracy, r2, precision)

cm = confusion_matrix(y_test, y_hat, labels=[1.0, 0.0])
print(cm)
matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
matrix.plot()
matrix.ax_.set_title('Confusion Matrix', color='black')
plt.xlabel('Predicted Subtype', color='black')
plt.ylabel('Actual Subtype', color='black')
plt.gcf().axes[0].tick_params(colors='black')
plt.gcf().axes[1].tick_params(colors='black')
plt.gcf().set_size_inches(10, 6)
plt.rcParams['font.size'] = '30'

plt.show()

y_scores = model.predict_proba(x_test)[:, 1]
print(average_precision_score(y_test, y_scores))
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
print(precision, recall)
'''
#create precision recall curve
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')

#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

#display plot
plt.show()'''