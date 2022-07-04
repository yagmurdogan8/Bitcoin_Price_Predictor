# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow import keras
from tensorflow.keras import layers

dataBTC = pd.read_csv('BTC-USD.csv', date_parser = True)
dataBTC.tail()

Test_dataBTC = dataBTC[dataBTC['Date']> '2020-01-08'].copy()
Test_dataBTC

training_dataBTC = data_training.drop(['Date', 'Adj Close'], axis = 1)
training_dataBTC.head()

scaler = MinMaxScaler()
training_dataBTC = scaler.fit_transform(training_dataBTC)
training_dataBTC

X_train = []
Y_train = []

training_dataBTC.shape[0]

for i in range(90, training_dataBTC.shape[0]):
    X_train.append(training_dataBTC[i-90:i])
    Y_train.append(training_dataBTC[i,0])

X_train, Y_train = np.array(X_train), np.array(Y_train)

X_train.shape



model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))
model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units =1))

model.summary()

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

history= model.fit(X_train, Y_train, epochs = 20, batch_size =50, validation_split=0.1)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title("Training and Validation Loss")
plt.legend()

plt.show()

Last_90_Days = data_training.tail(90)
df= Last_90_Days .append(Test_dataBTC, ignore_index = True)
df = df.drop(['Date', 'Adj Close'], axis = 1)
df.head()

inputs = scaler.transform(df)
inputs

X_test = []
Y_test = []
for i in range (90, inputs.shape[0]):
    X_test.append(inputs[i-90:i])
    Y_test.append(inputs[i, 0])

X_test, Y_test = np.array(X_test), np.array(Y_test)
X_test.shape, Y_test.shape

Y_pred = model.predict(X_test)
Y_pred, Y_test

scaler.scale_

scale = 1/5.18164146e-05
scale

Y_test = Y_test*scale
Y_pred = Y_pred*scale

Y_pred

Y_test

plt.figure(figsize=(14,5))
plt.plot(Y_test, color = 'red', label = 'Real Bitcoin Price')
plt.plot(Y_pred, color = 'black', label = 'Our Predicted Bitcoin Price')
plt.title('Bitcoin Price Predictor by Yağmur DOĞAN & Batuhan KESİKBAŞ')
plt.xlabel('Day')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()





