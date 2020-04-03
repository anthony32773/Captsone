import requests
import json
import csv
import os
import pandas as pd
import datetime
import plotly
import numpy as np
import tensorflow as tf
from keras import models
from keras import optimizers
from keras import losses
from keras import activations
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras import utils
from keras.layers import LSTM
from keras.layers import Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.python.client import device_lib

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["192.168.0.37:12345", "192.168.0.43:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})

multiWorkerStrategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

alphaAPIKey = "ADEN9ZLHN6SZGYKX"

apiLink = "https://www.alphavantage.co/query?"

apiOptions = "function=TIME_SERIES_DAILY&symbol=AAPL&outputsize=full&datatype=json&apikey=" + alphaAPIKey

response = requests.get(apiLink + apiOptions)

response = json.loads(response.content.decode('utf-8'))

response = response['Time Series (Daily)']

df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
for d, p in response.items():
    date = datetime.datetime.strptime(d, '%Y-%m-%d').date()
    data_row = [date, float(p['1. open']), float(p['2. high']), float(p['3. low']), float(p['4. close']), int(p['5. volume'])]
    df.loc[-1,:] = data_row
    df.index = df.index + 1
df = df.sort_values('date', ascending=True)

timeSteps = 60
batchSize = 20
df.drop(columns=['open', 'high', 'low', 'volume'], inplace=True)

# plt.figure()
# plt.plot(df['date'], df['close'])
# plt.title('Test')
# plt.ylabel('Price')
# plt.xlabel('Days')
# plt.show()

closeData = df['close'].values
closeData = closeData.reshape((-1, 1))

dates = df['date'].values

# Train = 70%, Validation = 20%, Test = 10%

closeTrain, closeTest = train_test_split(closeData, train_size=0.7, test_size=0.3, shuffle=False)
closeVal, closeTestActual = train_test_split(closeTest, train_size=0.7, test_size=0.3, shuffle=False)

datesTrain, datesTest = train_test_split(dates, train_size=0.7, test_size=0.3, shuffle=False)
datesVal, datesTestActual = train_test_split(datesTest, train_size=0.7, test_size=0.3, shuffle=False)

min_max_scaler = MinMaxScaler(feature_range=(0, 1))
closeTrain = min_max_scaler.fit_transform(closeTrain)
closeTestActual = min_max_scaler.transform(closeTestActual)
closeVal = min_max_scaler.transform(closeVal)

trainGenerator = TimeseriesGenerator(closeTrain, closeTrain, length=timeSteps, batch_size=10)
valGenerator = TimeseriesGenerator(closeVal, closeVal, length=timeSteps, batch_size=1)
testGenerator = TimeseriesGenerator(closeTestActual, closeTestActual, length=timeSteps, batch_size=1)

x, y = trainGenerator[0]
# print(f"{x} -> {y}")
# for i in trainGenerator:
#     print(i)
# print(trainGenerator)

# def build_model():
#     model = Sequential()
#     model.add(LSTM(10, activation='relu', input_shape=(timeSteps, 1)))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mse')
#     return model

def build_model():
    model = models.Sequential()
    model.add(LSTM(100, activation=activations.relu(), input_shape=(timeSteps, 1)))
    model.add(Dropout(1))
    #model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(1, activation=activations.linear()))
    print("made it2")
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss=tf.keras.losses.mse, metrics=['acc'])
    return model

    # priceModel = Sequential()
    # priceModel.add(LSTM(units=100, return_sequences=True,
    #                input_shape=(timeSteps, 1)))
    # priceModel.add(Dropout(0.2))
    # priceModel.add(LSTM(units=100, return_sequences=True))
    # priceModel.add(Dropout(0.2))
    # priceModel.add(LSTM(units=100, return_sequences=True))
    # priceModel.add(Dropout(0.2))
    # priceModel.add(LSTM(units=100, return_sequences=False))
    # priceModel.add(Dropout(0.2))
    # priceModel.add(Dense(units=1))
    # priceModel.compile(optimizer='adam', loss='mse', metrics=['acc'])

NUM_EPOCHS = 10
VERBOSE = 1
STEPS = len(trainGenerator)

with multiWorkerStrategy.scope():
    model = models.Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(timeSteps, 1)))
    model.add(Dropout(1))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=losses.mse, metrics=['accuracy'])

# history = model.fit(trainGenerator, epochs=10)
history = model.fit(trainGenerator, epochs=NUM_EPOCHS, verbose=VERBOSE, validation_data=valGenerator, steps_per_epoch=STEPS)

results = model.evaluate_generator(testGenerator)
print(results)

# priceModel.save('myModel.h5')

# priceModel = load_model('myModel.h5')

# These are all metrics that are recorded while training
# history_dict = history.history
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# acc = history_dict['acc']
#
# epochs = range(1, len(acc) + 1)
#
# # Plotting training and validation loss
# plt.plot(epochs, loss_values, 'bo', label='Training Loss') # bo = blue dot
# plt.plot(epochs, val_loss_values, 'b', label='Validation Loss') # b = solid blue line
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# plt.clf()
# val_acc_values = history_dict['val_acc']
# plt.plot(epochs, acc, 'bo', label='Training Accuracy')
# plt.plot(epochs, val_acc_values, 'b', label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

#priceModel = load_model('myModel.h5')


closeData = closeData.reshape((-1))
predList = []
test = np.empty(60)

for i in range(0, 4):
    print(i)
    test = test.reshape((-1))
    test = closeData[-timeSteps + i:]
    for j in range(0, i):
        test = np.append(test, predList[j])
    print(test)
    test2 = test.reshape((-1, 1))
    test2 = min_max_scaler.transform(test2)
    test2 = test2.reshape((1, timeSteps, 1))
    output = model.predict(test2)
    output = min_max_scaler.inverse_transform(output)
    predList.append(output[0][0])
    print(f"Pred {i}: {output[0][0]}")

print(predList)