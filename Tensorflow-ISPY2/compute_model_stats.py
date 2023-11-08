import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import yaml
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

drug = 'Paclitaxel+MK-2206'
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
'''
params = yaml.safe_load(param_input_path)

model = train_eval_model(num_layers=params['num_layers'],
                         num_epochs=params['num_epochs'],
                         batch_size=params['batch_size'],
                         lr=params['lr'],
                         layer_activation=params['layer_activation'],
                         output_activation=params['output_activation'],
                         dropout=params['dropout'],
                         save=True, plot=True)
'''
y_pred = model.predict(x_test)
y_pred_c = [int(y_pred[i]>0.5) for i in range(len(y_pred))]
'''
print('Precision: %.3f' % precision_score(y_test, y_pred_c))
print('Recall: %.3f' % recall_score(y_test, y_pred_c))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred_c))
print('F1 Score: %.3f' % f1_score(y_test, y_pred_c))
'''

print(y_pred_c)

cm = confusion_matrix(y_test, y_pred_c, labels=[1,0])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,0])
disp.plot(cmap='plasma')
plt.rcParams.update({'font.size': 30})
plt.savefig('DrugData/' + drug + '/' + drug + "_confusion_matrix.png")
plt.show()