#%%
#importing...
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
Scaler = MinMaxScaler(feature_range=(0,1))

from sklearn.linear_model import LinearRegression


#imports for model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error,accuracy_score


class LSTMPrediction:
    
    def __init__(self,symbol,look_back):
        self.symbol = symbol
        self.timeframe = look_back

    def fetchFromYahoo(self):
        yobj = yf.Ticker(self.symbol)
        tickerDict = yobj.info
        #print(yobj.info.keys())
        df = yobj.history(period=self.timeframe)
        df = df.drop(['Stock Splits','Dividends'],axis=1)
        df.index =  pd.to_datetime(df.index)
        #print('\n'+tickerDict['longBusinessSummary'])
        print(df.tail())
        plt.plot(df['Close'])
        return df,tickerDict
       
    
    def get_train_test_dataset(self,df,training_size=0.70,testing_size=0.30):
        try:
            print('this will return a training and test data')
            print('\n'+'Recent Data' + '\n',df.tail())
            print('MEAN CLOSE: ',df['Close'].mean())
            print('MAX CLOSE: ',df['Close'].max())
            print('MIN CLOSE: ',df['Close'].min())
            close_price = df.reset_index()['Close']
            close_price = Scaler.fit_transform(np.array(close_price).reshape(-1,1))
            train_size = int(len(close_price)*training_size)
            test_size = int(len(close_price*testing_size))
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


    def create_LSTM_model(self,lstm_layers_after_main=0,lstm_units=32,shape=(),loss='mean_squared_error',optimizer='adam'):
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




class LinearRegPrediction:

    def get_preds_lin_reg(self, df, target_col='Close'):
       
        regressor = LinearRegression()
        x = df.drop(target_col, axis=1)
        y = df[target_col]
        xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.1, random_state=0)
        
        regressor.fit(xtrain, ytrain)
        y_pred = regressor.predict(xtest)
        ytest = np.array(ytest).reshape(-1,1)
        y_pred = np.array(y_pred).reshape(-1,1)
        print(regressor.score(ytest,y_pred))
        #pred_min = min(y_pred)
        #print(pred_min)
        valid = pd.DataFrame()
        valid['Valid'] = ytest
        valid['Prediction'] = y_pred
        print('Standard Deviation: ',np.std(y_pred))
        print('RMSE: ' , np.sqrt(mean_squared_error(ytest,y_pred)))
        

class Technicals:
    def __init__(self,symbol):
        self.symbol = symbol

    def EMA(self,timeframe=9,on_field='Close'):
        yobj = yf.Ticker(self.symbol)
        df = yobj.history(period="1y")
        df = df.drop(['Stock Splits','Dividends'],axis=1)
        df.index =  pd.to_datetime(df.index)
        EMA = df[on_field].ewm(span=timeframe, adjust=False).mean()
        df_new = df[[on_field]]
        df_new.reset_index(level=0, inplace=True)
        df_new.columns=['ds','y']
        plt.figure(figsize=(16,8))
        plt.plot(df_new.ds, df_new.y, label='price')
        plt.plot(df_new.ds, EMA, label='EMA line',color='red')
        plt.show()
        print('Latest EMA on '+on_field+': ',EMA[len(EMA)-1],'\n')
        return EMA


    def MACD(self,on_field='Close'):
        yobj = yf.Ticker(self.symbol)
        df = yobj.history(period="1y")
        df = df.drop(['Stock Splits','Dividends'],axis=1)
        df.index =  pd.to_datetime(df.index)
        df_new = df[[on_field]]
        df_new.reset_index(level=0, inplace=True)
        df_new.columns=['ds','y']
        #df_new.head()  
        EMA12 = df_new.y.ewm(span=12, adjust=False).mean()
        EMA26 = df_new.y.ewm(span=26, adjust=False).mean()
        MACD = EMA12-EMA26
        EMA9 = MACD.ewm(span=9, adjust=False).mean()
        plt.figure(figsize=(16,8))
        #plt.plot(df_new.ds, df_new.y, label='price')
        plt.plot(df_new.ds, MACD, label=self.symbol+' MACD', color='blue')
        plt.plot(df_new.ds, EMA9, label=self.symbol+' Signal Line', color='red')
        plt.legend(loc='upper left')
        plt.show()
        print('\n')
        print(EMA9[len(EMA9)-1], MACD[len(MACD)-1])
        return EMA9[len(EMA9)-1], MACD[len(MACD)-1]  #latest value of EMA9 line and MACD value
        

    def RSI(self, period = 14):
        # If the RSI value is over 70, the security is considered overbought, if the value is lower than 30,
        # it is considered to be oversold
        # Using a conservative approach, sell when the RSI value intersects the overbought line
        # buy when the value intersects the oversold line (for blue chip stocks)
        yobj = yf.Ticker(self.symbol)
        df = yobj.history(period="1y")
        df = df.drop(['Stock Splits','Dividends'],axis=1)
        df_index =  pd.to_datetime(df.index)
        
        change = []
        gain = []
        loss = []
        AvgGain = []
        AvgLoss = []
        RS = []
        RSI = []
        df_new = pd.DataFrame(df['Close'], index=df.index)
        change.insert(0,0)
        #change calc
        for i in range(1,len(df_new)):
            diff = df_new.Close[i] - df_new.Close[i-1]
            change.append(diff)  
        df_new['Change'] = change
        #Gain and loss
        for i in range(len(df_new)):
            if df_new.Change[i] > 0:
                gain.append(df_new.Change[i])
                loss.append(0)
            elif df_new.Change[i] < 0:
                loss.append(abs(df_new.Change[i]))
                gain.append(0)
            else:
                gain.append(0)
                loss.append(0)
        
        df_new['Gain'] = gain
        df_new['Loss'] = loss

        #average gain/loss
        averageSum_forgain = 0
        averageSum_forloss = 0
        averageGain = 0
        averageLoss = 0
        
        count = 1
        for i in range(0,len(df_new)):
            averageSum_forgain = averageSum_forgain + df_new.Gain[i]
            averageGain = averageSum_forgain/count
            AvgGain.insert(i,round(averageGain,4))    
            averageSum_forloss = averageSum_forloss + df_new.Loss[i]
            averageLoss = averageSum_forloss/count
            AvgLoss.insert(i,round(averageLoss,4))
            count+=1

            if averageGain == 0 or averageLoss == 0:
                RS.append(0.0)
            else:
                RS.append(averageGain/averageLoss)    
            
        
        df_new['AvgGain'] = AvgGain
        df_new['AvgLoss'] = AvgLoss
        df_new['RS'] = RS
        rsi = 0
        for i in range(0,len(df_new)):
            rsi = 100 - 100/(1+df_new.RS[i])
            RSI.append(round(rsi,2))

        df_new['RSI'] = RSI
        
        
        plt.figure(figsize=(16,8))
        plt.plot(df_index[len(df_new)-period:len(df_new)],df_new.iloc[len(df_new)-period:len(df_new),-1], label='RSI value')
        plt.legend(loc='upper left')
        plt.show()
        print('\nCurrent RSI value: ' , df_new['RSI'][-1])
        Latest_RSI_value = float(df_new['RSI'][-1])
        return df_new, Latest_RSI_value

    def BollingerBands(self, degree_of_freedom = 20, period = 20, on_field = 'Close'):    
        yobj = yf.Ticker(self.symbol)
        df = yobj.history(period="1y")
        df = df.drop(['Stock Splits','Dividends'],axis=1)
        df_index =  pd.to_datetime(df.index)
        #print(df[on_field].rolling(window = period).sum()/period)
        #SMA calculated
        MA = df[on_field].rolling(window = period).sum()/period
        typical_price = []       
        #printing SMA
        
        
        #printing BOLU
        BOLU = []
        BOLD = []
        for i in range(len(df)-period,len(df)):
            #typical price = (high+low+close)/3
            typical_price.append((df.iloc[i,1] + df.iloc[i,2] + df.iloc[i,3]) / 3)
        
        typical_price = pd.Series(typical_price)

        for i in range(len(typical_price)):
            std = 2*(    math.sqrt(  math.pow(i-typical_price.mean(),2) / len(typical_price) )    )   
            BOLU.append(typical_price[i] + std)
            BOLD.append(typical_price[i] - std)

        # BOLU = pd.Series(BOLU)
        # BOLD = pd.Series(BOLD)

        print("Middle value: " + str(MA.iloc[-1]))
        print("Upper Band: " + str(BOLU[-1]))
        print("Lower Band: " + str(BOLD[-1]))


    
        

    
        



class OptionChainAnalysis:
    def __init__(self,view_options_contract_for,symbol,expiry_date,strike_price):
        super().__init__()
    



#%%

#obj = LSTMPrediction('PNB.NS')
#df, dictionary = obj.fetchFromYahoo()


obj2 = Technicals('NMDC.NS')
#EMA = obj2.EMA(20)
obj2.MACD()
df_new, RSI = obj2.RSI()
d =obj2.BollingerBands()
#%%


#%%

# train_data, test_data = obj.get_train_test_dataset(df)

# xtrain, ytrain = obj.prepare_data_for_LSTM_kaggle(train_data)
# xtest, ytest = obj.prepare_data_for_LSTM_kaggle(test_data)


# xtrain, xtest = obj.reshape_for_LSTM(xtrain,xtest)

# model = obj.create_LSTM_model(lstm_layers_after_main=2,
#                                 lstm_units=32,
#                                 shape=(xtrain.shape[1],1)
#                                 )


# model.fit(xtrain,ytrain,batch_size=16,epochs=10,validation_data=(xtest,ytest))

# predictions = model.predict(xtest)
# predictions_inverse = Scaler.inverse_transform(predictions)
# rmse = np.sqrt(np.mean((predictions - ytest) ** 2))
# rmse

# # Plot the data
# train_data_len = len(train_data)

# index_for_date = pd.DataFrame(df.index)
# train = pd.DataFrame(data=Scaler.inverse_transform(train_data),columns={'Close'})
# train_size = int(len(df['Close'])*0.70)

# valid = pd.DataFrame(data=Scaler.inverse_transform(test_data[60:,:]),columns={'Close'})
# valid['Predictions'] = predictions_inverse

# # Visualize the data
# plt.figure(figsize=(16,8))
# plt.title('Model')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price', fontsize=18)
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])
# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
# plt.show()

# print(valid)

#%%
import math
import numpy as np
a = [12,23,34,542,123,4321,12,34,6,7,9863,2,4,65,3,745,74]
a = np.array(a)
std_for_a = []
for i in a:
    std = math.sqrt(((i-a.mean())**2)/len(a))
    std_for_a.append(std)

print(std_for_a)

# %%
import numpy as np
a = [1,2,3,4]

np.array(a).sum()
# %%
