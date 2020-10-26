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
        scaler = MinMaxScaler()
        close_price = scaler.fit_transform(np.array(close_price).reshape(-1,1))
        train_size = int(len(close_price)*0.70)
        test_size = int(len(close_price*0.30))
        train_data = close_price[0:train_size,:]
        test_data = close_price[train_size:len(close_price),:1]
        return train_data,test_data
    except ValueError:
        print('Try a different scrip')
# %%

train_data, test_data = dailyOnClose('BSE:SBIN')
# %%
def create_dataset(dataset, timestep=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - timestep-1):
        record = dataset[i:(i+timestep),0]
        dataX.append(record)
        dataY.append(dataset[i + timestep, 0])
    return np.array(dataX), np.array(dataY)
# %%
xtrain, ytrain = create_dataset(train_data,timestep=100)
xtest, ytest = create_dataset(test_data, timestep=100)
# %%
xtrain = xtrain.reshape(xtrain.shape[0],xtrain.shape[1], 1)
xtest = xtest.reshape(xtest.shape[0],xtest.shape[1], 1)
# %%
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(xtrain.shape[1],xtrain.shape[2])))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.summary()

# %%
model.fit(xtrain,ytrain,batch_size=64,epochs=20,validation_data=(xtest,ytest),verbose=0)

# %%
