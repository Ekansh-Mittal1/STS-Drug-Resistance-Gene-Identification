import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.layers import Input, Dense, BatchNormalization, LeakyReLU
from keras.models import Model
from keras.utils import plot_model

drug = 'Paclitaxel+Pembrolizumab'
df_path = "DrugData/" + drug + "/" + drug + "_scaled_dataset.tsv"
encoder_output = "DrugData/" + drug + "/" + drug + "_encoder.h5"

# load data
df = pd.read_csv(df_path, delimiter='\t')

# split data
y_df = df[['TxResponse']].copy()
x_df = df.drop('TxResponse', axis=1)

# gene list
genes = x_df.columns

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=42)

#https://machinelearningmastery.com/autoencoder-for-classification/

n_inputs = 7000

# define encoder
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs*2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = n_inputs/10
bottleneck = Dense(n_bottleneck)(e)

# define decoder, level 1
d = Dense(n_inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(n_inputs*2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(n_inputs, activation='linear')(d)
# define autoencoder model
model = Model(inputs=visible, outputs=output)

#compile
model.compile(optimizer='adam', loss='mse')

# plot the autoencoder
plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)

#fit
history = model.fit(x_train, x_train, epochs=200, verbose=2, validation_data=(x_test,x_test))

# plot loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
#plt.show()
# define an encoder model (without the decoder)
encoder = Model(inputs=visible, outputs=bottleneck)
encoder.compile(optimizer='adam', loss='mse')
plot_model(encoder, 'encoder_compress.png', show_shapes=True)
# save the encoder to filev
#encoder.save(encoder_output)

post = model.predict(x_test)

a = np.array(x_test) # your x
b = np.array(post) # your y
mses = ((a-b)**2).mean(axis=1)
print(mses)
