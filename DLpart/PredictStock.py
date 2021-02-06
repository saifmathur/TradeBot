#%%
#imports
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
Scaler = MinMaxScaler(feature_range=(0,1))



#imports for model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


import math
from sklearn.metrics import mean_squared_error,accuracy_score

class LSTMPrediction:
    
    def __init__(self,symbol):
        self.symbol = symbol
        

    def fetchFromYahoo(self):
        yobj = yf.Ticker(self.symbol)
        tickerDict = yobj.info
        #print(yobj.info.keys())
        df = yobj.history(period="max")
        df = df.drop(['Dividends','Stock Splits'],axis=1)
        df.index =  pd.to_datetime(df.index)
        #print('\n'+tickerDict['longBusinessSummary'])
        print(df.tail())
        plt.plot(df['Close'])
        return df,tickerDict
       
    
    def get_train_test_dataset(self,df):
        try:
            print('this will return a training and test data')
            print('\n'+'Recent Data' + '\n',df.tail())
            print('MEAN CLOSE: ',df['Close'].mean())
            print('MAX CLOSE: ',df['Close'].max())
            print('MIN CLOSE: ',df['Close'].min())
            close_price = df.reset_index()['Close']
            close_price = Scaler.fit_transform(np.array(close_price).reshape(-1,1))
            train_size = int(len(close_price)*0.70)
            test_size = int(len(close_price*0.30))
            train_data = close_price[0:train_size,:]
            test_data = close_price[train_size:len(close_price),:1]
            return train_data,test_data
        except ValueError:
            print('Try a different Scrip')


    def prepare_data_for_LSTM_krish(self,dataset,timestep=1):
        dataX, dataY = [], []
        for i in range(len(dataset)- timestep-1):
            record = dataset[i:(i+timestep),0]
            dataX.append(record)
            dataY.append(dataset[i + timestep, 0])
        return np.array(dataX), np.array(dataY)


    def prepare_data_for_LSTM_kaggle(self,dataset):
        dataX = []
        dataY = []
        for i in range(60, len(dataset)):
            dataX.append(dataset[i-60:i, 0])
            dataY.append(dataset[i, 0])
            if i<=61 :
                print(dataX)
                print(dataY)
                print()

        dataX, dataY = np.array(dataX), np.array(dataY)
        return dataX, dataY


    def reshape_for_LSTM(self,train_data, test_data):
        train_data = train_data.reshape(train_data.shape[0],train_data.shape[1],1)
        test_data = test_data.reshape(test_data.shape[0],test_data.shape[1],1)
        return train_data, test_data


    def create_LSTM_model(self,lstm_layers_after_main=0,lstm_units=1,shape=(),loss='mean_squared_error',optimizer='adam'):
        dropout = 0.0
        model = Sequential()
        model.add(LSTM(lstm_units,return_sequences=True,input_shape=shape))
        if lstm_layers_after_main > 2 and lstm_layers_after_main < 5:
            dropout = 0.4
        elif lstm_layers_after_main <= 2:
            dropout = 0.1
        for i in range(lstm_layers_after_main):
            model.add(LSTM(lstm_units,return_sequences=True))
            if i % 2 == 0:
                continue
            model.add(Dropout(dropout))
        
        model.add(LSTM(lstm_units))
        model.add(Dense(1))
        print('Dropping out ' + str(dropout*100) + '%')
        model.summary()
        model.compile(loss=loss,optimizer=optimizer)
        return model




class MovingAveragePrediction:
    def __init__(self, symbol):
        self.symbol = symbol

    


#%%
symbol = str.upper(str(input('ENTER SYMBOL: ')))+'.NS'
obj = LSTMPrediction(symbol)
df,dictionary = obj.fetchFromYahoo()
train_data, test_data = obj.get_train_test_dataset(df)



xtrain, ytrain = obj.prepare_data_for_LSTM_kaggle(train_data)
xtest, ytest = obj.prepare_data_for_LSTM_kaggle(test_data)


xtrain, xtest = obj.reshape_for_LSTM(xtrain,xtest)

model = obj.create_LSTM_model(lstm_layers_after_main=2,
                                lstm_units=64,
                                shape=(xtrain.shape[1],1)
                                )


model.fit(xtrain,ytrain,batch_size=16,epochs=10,validation_data=(xtest,ytest))



#%%
predictions = model.predict(xtest)
predictions_inverse = Scaler.inverse_transform(predictions)


# %%
rmse = np.sqrt(np.mean((predictions - ytest) ** 2))
# %%
# Plot the data
train_data_len = len(train_data)
train = pd.DataFrame(data=Scaler.inverse_transform(train_data),columns={'Close'})
valid = pd.DataFrame(data=Scaler.inverse_transform(test_data[60:,:]),columns={'Close'})
valid['Predictions'] = predictions_inverse
# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
# %%
print(valid)
# %%
