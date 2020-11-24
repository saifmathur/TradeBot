# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.models import Sequential
from FetchData import fetchWeekly, fetchDaily, fetchIntraday, getFundamentalData
from key import KEY
from alpha_vantage.timeseries import TimeSeries
# ts = TimeSeries(key=KEY, output_format='pandas')
# data, meta_data = ts.get_daily(symbol='BSE:SBIN', outputsize='full')
from alpha_vantage.timeseries import TimeSeries
from key import KEY
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from nsetools.nse import Nse
import numpy as np

def dailyOnClose(symbol):
    ts = TimeSeries(key=KEY, output_format='pandas')
    try:
        dataframe,meta_data = ts.get_daily(symbol,outputsize='full')
        dataframe = dataframe.rename(columns={
        '1. open':'Open',
        '2. high':'High',
        '3. low':'Low',
        '4. close':'Close',
        '5. volume':'Volume'
    })
        dataframe = dataframe[['Open','High','Low','Close','Volume']]
        dataframe = dataframe.dropna()
        print('DATA INSIGHT \n')
        print(dataframe.head())
        print(dataframe.shape)
        print('MEAN CLOSE: ',dataframe['Close'].mean())
        print('MAX CLOSE: ',dataframe['Close'].max())
        print('MIN CLOSE: ',dataframe['Close'].min())
        print('/n')
        print('PREPROCESSING...')
        close_price = dataframe.reset_index()['Close']
        close_price = scaler.fit_transform(np.array(close_price).reshape(-1,1))
        train_size = int(len(close_price)*0.70)
        test_size = int(len(close_price*0.30))
        train_data = close_price[0:train_size,:]
        test_data = close_price[train_size:len(close_price),:1]
        return train_data,test_data
    except ValueError:
        print('Try a different scrip')

def create_dataset(dataset, timestep=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - timestep-1):
        record = dataset[i:(i+timestep),0]
        dataX.append(record)
        dataY.append(dataset[i + timestep, 0])
    return np.array(dataX), np.array(dataY)


try:
    train_data, test_data = dailyOnClose('')
    xtrain, ytrain = create_dataset(train_data,timestep=100)
    xtest, ytest = create_dataset(test_data, timestep=100)
    xtrain = xtrain.reshape(xtrain.shape[0],xtrain.shape[1], 1)
    xtest = xtest.reshape(xtest.shape[0],xtest.shape[1], 1)
    model = Sequential()
    model.add(LSTM(100,return_sequences=True, input_shape=(xtrain.shape[1],xtrain.shape[2])))
    model.add(LSTM(100,return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(100))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    model.fit(xtrain,ytrain,batch_size=64,epochs=20,validation_data=(xtest,ytest),verbose=1)

except TypeError:
    print('Check scrip name')
    



# %%
train_pred = model.predict(xtrain)
test_pred = model.predict(xtest)
train_pred = scaler.inverse_transform(train_pred)
test_pred = scaler.inverse_transform(test_pred)
# %%
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(ytest,test_pred))

# %%
# look_back = 100
# trainPredictPlot = np.empty_like(close_price)
# %%
plt.plot(test_pred)
plt.show()
# %%
