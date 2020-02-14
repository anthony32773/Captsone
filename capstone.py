import requests
import json
import csv
import pandas as pd
import datetime
from itertools import chain
import numpy as np
import quandl
from keras import models
from keras import layers
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm._tqdm_notebook import tqdm_notebook
from sklearn.metrics import mean_squared_error
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

quandl.ApiConfig.api_key = "a3nm_befbydzTL3UomTn"

api2Link = "https://www.quandl.com/api/v3/datasets/WIKI/AAPL/data.csv"

alphaAPIKey = "ADEN9ZLHN6SZGYKX"

apiLink = "https://www.alphavantage.co/query?"

apiOptions = "function=TIME_SERIES_DAILY&symbol=AAPL&outputsize=full&datatype=json&apikey=" + alphaAPIKey

response2 = requests.get(api2Link)

print(response2.content)

#print(json.dumps(response2, indent=2, sort_keys=True))

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
#print (df)

df.to_csv("test.csv")

datePrice = {}
dates = []

for i in df['close']:
    dates.append(i)

closingPrices = np.asarray(dates)

print (closingPrices)
print (closingPrices.shape[0])

trainingSet = closingPrices[0:3033]

validationSet = closingPrices[3033:4033]

print (validationSet)

testSet = closingPrices[4033:5033]

print (testSet)

# plt.figure()
# plt.plot(df['close'])
# plt.ylabel('Price $USD')
# plt.title('Apple Stock Price History')
# plt.xlabel('Days')
# plt.legend(['Close'], loc='upper left')
# plt.show()

train_cols = ["open", "high", "low", "close", "volume"]
df_train, df_test = train_test_split(df, train_size=0.8, test_size=0.2, shuffle=False)
print(f"Training Set Size: {len(df_train)}\nTesting Set Size: {len(df_test)}")
# Scale these between min max:
x = df_train.loc[:, train_cols].values
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(df_test.loc[:, train_cols])

TIMESTEPS = 60
BATCHSIZE = 20

def buildTimeSeries(input, yColIndex):
    dim0 = input.shape[0] - TIMESTEPS
    dim1 = input.shape[1]
    x = np.zeros((dim0, TIMESTEPS, dim1))
    y = np.zeros((dim0,))

    for i in range(dim0):
        x[i] = input[i:TIMESTEPS + i]
        y[i] = input[TIMESTEPS + i, yColIndex]
    print (f"Length of Time Series:\nX: {x.shape}\nY: {y.shape}")
    return x, y

def trimDataSet(input, batchSize):
    rowsDrop = input.shape[0] % batchSize
    if rowsDrop > 0:
        print (f"Number of rows dropped: {rowsDrop}")
        return input[:-rowsDrop]
    else:
        return input

xT, yT = buildTimeSeries(x_train, 3)
xT = trimDataSet(xT, BATCHSIZE)
yT = trimDataSet(yT, BATCHSIZE)
xTemp, yTemp = buildTimeSeries(x_test, 3)
xVal, x_test_t = np.split(trimDataSet(xTemp, BATCHSIZE), 2)
yVal, y_test_t = np.split(trimDataSet(yTemp, BATCHSIZE), 2)

def build_model():
    model = models.Sequential()
    model.add(LSTM(100, batch_input_shape=(BATCHSIZE, TIMESTEPS, xT.shape[2]), dropout=0.0, recurrent_dropout=0.0, stateful=True, kernel_initializer='random_uniform'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='mse')
    return model

priceModel = build_model()

history = priceModel.fit(xT, yT, epochs=50, batch_size=BATCHSIZE,
                         shuffle=False, validation_data=(trimDataSet(xVal, BATCHSIZE),
                                                         trimDataSet(yVal, BATCHSIZE)))

yPred = priceModel.predict(trimDataSet(x_test_t, BATCHSIZE), batch_size=BATCHSIZE)
yPred = yPred.flatten()
y_test_t = trimDataSet(y_test_t, BATCHSIZE)
error = mean_squared_error(y_test_t, yPred)
print(f"Error is: {error}, {yPred.shape}, {y_test_t.shape}")
print(yPred[0:15])
print(y_test_t[0:15])

y_pred_org = (yPred * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
y_test_t_org = (y_test_t * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]

plt.figure()
plt.plot(y_pred_org)
plt.plot(y_test_t_org)
plt.title('Prediction vs Real Stock Price')
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend(['Prediction', 'Real'], loc='upper left')
plt.show()



#
# model = build_model()
# num_epochs = 100
#
# history = model.fit(trainingSet, validation_data=validationSet, epochs=num_epochs, batch_size=1516)