import requests
import json
import pandas as pd
import datetime
import plotly
import plotly.graph_objs as go
import numpy as np
from keras import models
from keras.models import load_model
from keras import layers
from keras.layers import LSTM
from keras.layers import Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

def build_model():
    model = models.Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(timeSteps, 1)))
    model.add(Dropout(1))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['acc'])
    return model

NUM_EPOCHS = 10
VERBOSE = 1
STEPS = len(trainGenerator)

model = build_model()

history = model.fit_generator(trainGenerator, epochs=NUM_EPOCHS, verbose=VERBOSE, validation_data=valGenerator, steps_per_epoch=STEPS)

results = model.evaluate_generator(testGenerator)

model.save('myModel.h5')

# model = load_model('myModel.h5')

def checkWeekday(todaysDate):
    if todaysDate.weekday() == 5:
        todaysDate = todaysDate + datetime.timedelta(days=2)
    elif todaysDate.weekday() == 6:
        todaysDate = todaysDate + datetime.timedelta(days=1)
    return todaysDate

closeData = closeData.reshape((-1))
todayDate = datetime.datetime.now().date()
predDateList = []
predList = []
test = np.empty(60)

allDatesNP = df['date'].values
allClosingsNP = df['close'].values

allDatesList = allDatesNP.tolist()
allClosingsList = allClosingsNP.tolist()

for i in range(timeSteps, len(allClosingsNP)):
    test = test.reshape((-1))
    test = closeData[i - timeSteps:i]
    test2 = test.reshape((-1, 1))
    test2 = min_max_scaler.transform(test2)
    test2 = test2.reshape((1, timeSteps, 1))
    output = model.predict(test2)
    output = min_max_scaler.inverse_transform(output)
    predDateList.append(allDatesList[i])
    predList.append(round(output[0][0], ndigits=2))

numPreds = len(predList)

for i in range(0, 30):
    test = test.reshape((-1))
    test = closeData[-timeSteps + i:]
    for j in range(0, i):
        test = np.append(test, predList[j + numPreds - 1])
    test2 = test.reshape((-1, 1))
    test2 = min_max_scaler.transform(test2)
    test2 = test2.reshape((1, timeSteps, 1))
    output = model.predict(test2)
    output = min_max_scaler.inverse_transform(output)
    todayDate = todayDate + datetime.timedelta(days=1)
    todayDate = checkWeekday(todayDate)
    predDateList.append(todayDate)
    predList.append(round(output[0][0], ndigits=2))
    print(f"Pred {i}: {round(output[0][0], ndigits=2)}")

closeGraph = go.Scatter(
    x=allDatesList,
    y=allClosingsList,
    name='Historical Data'
)

predGraph = go.Scatter(
    x=predDateList,
    y=predList,
    name='Prediction'
)

layout = go.Layout(title='AAPL Closing Prices',
                   xaxis=dict(title='Date'),
                   yaxis=dict(title='Price')
                   )

fig = go.Figure(data=[closeGraph, predGraph], layout=layout)

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
plotly.io.write_html(fig, file='AAPLGraph.html', auto_open=True)