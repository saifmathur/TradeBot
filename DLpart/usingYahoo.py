# %%

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
import pandas as pd
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt



def fetchFromYahoo(symbol,**kwargs):
        yobj = yf.Ticker(symbol)
        tickerDict = yobj.info
        #print(yobj.info.keys())
        df = yobj.history(period="max")
        df = df.drop(['Dividends','Stock Splits'],axis=1)
        df.index =  pd.to_datetime(df.index)
        #print('\n'+tickerDict['longBusinessSummary'])
        print(df.tail())
        return df,tickerDict
        plt.plot()

#df,info = fetchFromYahoo('ABBOTINDIA.NS')
#tail = df.tail()
#close = tail.reset_index()['Close']

#print('MAX PRICE IN 5 DAYS: ',close.max())
#print('MIN PRICE IN 5 DAYS: ',close.min())
#plt.plot(close)




def get_train_test_dataset(df):
    try:
        print('this will return a training and test data')
        print('\n DATA INSIGHT')
        print('\n'+'Recent Data' + '\n',df.tail())
        print('Shape of dataframe',df.shape)
        print('MEAN CLOSE: ',df['Close'].mean())
        print('MAX CLOSE: ',df['Close'].max())
        print('MIN CLOSE: ',df['Close'].min())
        print('\n')
        print('PREPROCESSING...')
        close_price = df.reset_index()['Close']
        close_price = scaler.fit_transform(np.array(close_price).reshape(-1,1))
        train_size = int(len(close_price)*0.70)
        test_size = int(len(close_price*0.30))
        train_data = close_price[0:train_size,:]
        test_data = close_price[train_size:len(close_price),:1]
        return train_data,test_data
    except ValueError:
        print('Try a different Scrip')


# %%
def create_dataset(dataset,timestep=1):
    dataX, dataY = [], []
    for i in range(len(dataset)- timestep-1):
        record = dataset[i:(i+timestep),0]
        dataX.append(record)
        dataY.append(dataset[i + timestep, 0])
    return np.array(dataX), np.array(dataY)



# %%
#creating dataset 
df,info = fetchFromYahoo('BEPL.NS')
train_data, test_data = get_train_test_dataset(df)
xtrain, ytrain = create_dataset(train_data, timestep=100)
xtest, ytest = create_dataset(test_data, timestep=100)
# %%
xtrain = xtrain.reshape(xtrain.shape[0],xtrain.shape[1],1)
xtest = xtest.reshape(xtest.shape[0],xtest.shape[1],1)

# %%
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
model = Sequential()
model.add(LSTM(100,return_sequences=True, input_shape = (xtrain.shape[1],xtrain.shape[2])))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.summary()

# %%
model.fit(xtrain, ytrain, batch_size=64, epochs=20, validation_data=(xtest,ytest))
# %%

train_pred = model.predict(xtrain)
train_pred_inverse = scaler.inverse_transform(train_pred)


import matplotlib.pyplot as plt
plt.plot(train_pred)
plt.show()



# %%
model.save('BEPLtrain.h5')

# %%
test_pred = model.predict(xtest)
test_pred_inverse = scaler.inverse_transform(test_pred)
# %%
