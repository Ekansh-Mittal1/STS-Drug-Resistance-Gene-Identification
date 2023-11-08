import pandas as pd
import tensorflow as tf
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import math
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

drug = "Paclitaxel+MK-2206"
df_path = "../LogisticRegression-ISPY2/DrugData/" + drug + "/" + drug + "_scaled_dataset.tsv"
gene_scores = "../LogisticRegression-ISPY2/DrugData/" + drug + "/" + "chi2_scores.tsv"
param_output_path = "DrugData/" + drug + "/" + "params.yaml"
model_save_path = "DrugData/" + drug + "/optimized_model.h5"
num_genes = 7000
num_inputs = 7000

genes = pd.read_csv(gene_scores, delimiter='\t')['GENES'].tolist()[:num_genes] + ['TxResponse']
df = pd.read_csv(df_path, delimiter='\t', usecols=genes)

y_df = df[['TxResponse']].copy()
x_df = df.drop('TxResponse', axis=1)

smote = SMOTE(random_state=42, sampling_strategy="minority")
x_df, y_df = smote.fit_resample(x_df, y_df)

x_train, x_rem, y_train, y_rem = train_test_split(x_df, y_df, train_size=0.8, random_state=42)

x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size=0.5, random_state=42)

x_train.to_csv()
x_test.to_csv()
y_train.to_csv()
y_test.to_csv()

inputs = tf.keras.Input(shape=(num_inputs))
x = layers.Dense(num_inputs / 2, activation='relu')(inputs)
for i in range(1, 2):
    nodes = int(math.ceil(num_inputs / 2 ** i))
    x = layers.Dense(nodes, activation='relu')(x)

outputs = layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,
              loss='binary_crossentropy')
history = model.fit(x_train, y_train, epochs=200, verbose=1,
                    validation_data=(x_valid, y_valid))

y_pred = model.predict(x_test)
y_pred_c = [int(y_pred[i] > 0.5) for i in range(len(y_pred))]

print('Precision: %.3f' % precision_score(y_test, y_pred_c))
print('Recall: %.3f' % recall_score(y_test, y_pred_c))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred_c))
print('F1 Score: %.3f' % f1_score(y_test, y_pred_c))

cm = confusion_matrix(y_test, y_pred_c, labels=[1, 0])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 0])
disp.plot(cmap='plasma')
plt.rcParams.update({'font.size': 30})
plt.show()
