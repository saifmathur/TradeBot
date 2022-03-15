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

import sys
#sys.path.append('../DLpart/')
#from PredictStock import Technicals
import datetime

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

    def EMA(self,timeframe=9,on_field='Close',plot=False, period = "1y", interval = "1d"):
        df = yf.Ticker(self.symbol).history(period=period, interval=interval)
        df = df.drop(['Stock Splits','Dividends'],axis=1)
        df.index =  pd.to_datetime(df.index)
        EMA = df[on_field].ewm(span=timeframe, adjust=False).mean()
        df_new = df[[on_field]]
        df_new.reset_index(level=0, inplace=True)
        df_new.columns=['ds','y']
        if plot == True:
            plt.figure(figsize=(16,8))
            plt.plot(df_new.ds, df_new.y, label='price')
            plt.plot(df_new.ds, EMA, label='EMA line',color='red')
            plt.show()
        #print('Latest EMA on '+on_field+': ',EMA[len(EMA)-1],'\n')
        #return EMA
        return EMA[len(EMA)-1]


    def MACD(self,on_field='Close',plot=False):
        df = yf.Ticker(self.symbol).history(period="1y")
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
        
        #plt.plot(df_new.ds, df_new.y, label='price')
        if plot == True:
            plt.figure(figsize=(16,8))
            plt.plot(df_new.ds, MACD, label=self.symbol+' MACD', color='blue')
            plt.plot(df_new.ds, EMA9, label=self.symbol+' Signal Line', color='red')
            plt.legend(loc='upper left')
            plt.show()
        #print('\n')
        #print(EMA9[len(EMA9)-1], MACD[len(MACD)-1])
        if MACD[len(MACD)-1] > MACD[len(MACD)-2]:
            return True
        else:
            return False

        # if MACD[len(MACD)-1]-EMA9[len(EMA9)-1] <= 4 and MACD[len(MACD)-1]-EMA9[len(EMA9)-1] >= 0:  
        #     print('ALERT: MACD crossover about to occur, Sell side')
        # elif MACD[len(MACD)-1]-EMA9[len(EMA9)-1] >= -4 and MACD[len(MACD)-1]-EMA9[len(EMA9)-1] <= 0:
        #     print('ALERT: MACD crossover about to occur, Buy side')
        # else:
        #     print('No MACD crossovers')

        #return EMA9[len(EMA9)-1], MACD[len(MACD)-1]  #latest value of EMA9 line and MACD value
        

    def RSI_backUpCode(self, period = 14):
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

    def RSI(self,period = 14, plot = False):
        df = yf.Ticker(self.symbol).history(period="1y")
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
            AvgGain.insert(i,averageGain)    
            averageSum_forloss = averageSum_forloss + df_new.Loss[i]
            averageLoss = averageSum_forloss/count
            AvgLoss.insert(i,averageLoss)
            count+=1

            if averageGain == 0 or averageLoss == 0:
                RS.append(0.0)
            else:
                RS.append(averageGain/averageLoss)    
            
        
        df_new['AvgGain'] = AvgGain
        df_new['AvgLoss'] = AvgLoss
        df_new['RS'] = RS
        rsi = 0
        for i in range(len(df)-14,len(df)):
            rsi = 100 - 100/(1+df_new.RS[i])
            RSI.append(round(rsi,2))

        #df_new['RSI'] = RSI
        if plot == True:
            plt.figure(figsize=(16,8))
            plt.plot(df_index[len(df_new)-period:len(df_new)],RSI, label='RSI value')
            plt.legend(loc='upper left')
            plt.show()
            print('\nCurrent RSI value: ' , RSI[len(RSI)-1])
        Latest_RSI_value = RSI[-1]
        Previous_day_rsi_value = RSI[-2]
        if (Previous_day_rsi_value < Latest_RSI_value) and (Latest_RSI_value >= 40) and (Latest_RSI_value <= 60):
            return True
        else:
            return False
        #return df_new, RSI
        #return RSI
        #return Latest_RSI_value

    def BollingerBands(self, degree_of_freedom = 20, period = 20, on_field = 'Close'):    
        yobj = yf.Ticker(self.symbol)
        df = yobj.history(period="1mo")
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





#general analysis
class StockListAnalysis:
    def __init__(self):
        self.niftyColumns = ['SYMBOL','OPEN','HIGH','LOW','PREVCLOSE','LTP','TODAYS_CHANGE',
                            'CHANGE_%','VOLUME','VALUE','52WH','52WL','1Y_CHANGE%','1M_CHANGE%']
        self.niftySectorColumns = ['INDEX','CURRENT','%CHANGE','OPEN','HIGH','LOW','PREVCLOSE','PREVDAY','1W_CLOSE','1M_CLOSE','1Y_CLOSE',
                                '52WH','52WL','1Y_CHANGE%','1M_CHANGE%']
        try:
            self.nifty_100_data = pd.read_csv('../PreFedIndexData/MW-NIFTY-100-'+datetime.datetime.strftime(datetime.datetime.today(),"%d-%b-%Y")+'.csv', names=self.niftyColumns,header=0)
            self.nifty_sector_data = pd.read_csv('../PreFedIndexData/MW-All-Indices-'+datetime.datetime.strftime(datetime.datetime.today(),"%d-%b-%Y")+'.csv', names=self.niftySectorColumns, header=0)
        except FileNotFoundError:
            self.nifty_100_data = pd.read_csv('PreFedIndexData/MW-NIFTY-100-'+'15-Oct-2021'+'.csv', names=self.niftyColumns,header=0)
            self.nifty_sector_data = pd.read_csv('PreFedIndexData/MW-All-Indices-'+'15-Oct-2021'+'.csv', names=self.niftySectorColumns, header=0)


    def AnalyzeNiftySectors(self):
        print('\nBest Sectors to invest in right now...')
        #FILTERING NIFTY SECTORS ABOVE       
        NIFTY_SECTORS = pd.DataFrame(self.nifty_sector_data.iloc[[13,14,15,17,18,19,20,21,22,23,24,44,45],:].values, columns=self.niftySectorColumns)
        NIFTY_SECTORS = NIFTY_SECTORS.reset_index(drop=True)
        #NIFTY_SECTORS.columns
        Highest_1Y_return_sectors = []
        Highest_1M_return_sectors = []
        for i in range(len(NIFTY_SECTORS)):
            if float(NIFTY_SECTORS['1Y_CHANGE%'][i]) > 50:
                #print(NIFTY_SECTORS['INDEX'][i])
                Highest_1Y_return_sectors.append([NIFTY_SECTORS['INDEX'][i],NIFTY_SECTORS['1Y_CHANGE%'][i]])
            if float(NIFTY_SECTORS['1M_CHANGE%'][i]) > 10:
                #print(NIFTY_SECTORS['INDEX'][i])
                Highest_1M_return_sectors.append([NIFTY_SECTORS['INDEX'][i],NIFTY_SECTORS['1M_CHANGE%'][i]])
        return pd.DataFrame(Highest_1Y_return_sectors, columns=['SECTOR','365_DAY_RETURN%']) , pd.DataFrame(Highest_1M_return_sectors, columns=['SECTOR','30_DAY_RETURN%']) 

    def SwingTrade(self): 
        #FILTERING NIFTY 100
        nifty_filtered = []
        for i in range(1,len(self.nifty_100_data)):
            try:
                if (float(self.nifty_100_data['1Y_CHANGE%'][i])>50) and (float(self.nifty_100_data['1M_CHANGE%'][i])>5):
                    #self.nifty_100_data['1Y_CHANGE%'][i]
                    #print(self.nifty_100_data['SYMBOL'][i])
                    nifty_filtered.append([self.nifty_100_data['SYMBOL'][i],self.nifty_100_data['1M_CHANGE%'][i]])
            except:
                continue

        nifty_filtered = pd.DataFrame(nifty_filtered, columns = ['SYMBOL','1_MONTH_RETURN%'])
        #SUGGESTIONS
        suggestions = []
        print('\n Please wait this might take a few seconds... \n')
        for i in range(len(nifty_filtered)):
            try:
                LTP = round(float(yf.Ticker(nifty_filtered['SYMBOL'][i]+'.NS').history(period='1d')['Close'][-1]),ndigits=2)
                suggestions.append([nifty_filtered['SYMBOL'][i],nifty_filtered['1_MONTH_RETURN%'][i],[Technicals(nifty_filtered['SYMBOL'][i] + '.NS').RSI()],[Technicals(nifty_filtered['SYMBOL'][i] + '.NS').EMA(timeframe=50)<LTP],[Technicals(nifty_filtered['SYMBOL'][i] + '.NS').MACD()],LTP,round(Technicals(nifty_filtered['SYMBOL'][i] + '.NS').EMA(timeframe=20, interval= "60m"),ndigits=-1)])
            except:
                continue
        suggestions = pd.DataFrame(suggestions, columns = ['SYMBOL','1_MONTH_RETURN%','GOOD_RSI_VALUE','LTP_ABOVE_50_EMA','GOOD_MACD','LAST_TRADED_PRICE_₹','20_EMA']) #short cut for rupee symbol is ctrl+shift+4 
        #print('\n', suggestions)
        print('\nBest stocks to invest in, at the moment for swing trading...')
        best_for_swing_trade = []
        for i in range(len(suggestions)):
            if (suggestions['LTP_ABOVE_50_EMA'][i][0]==True) and (suggestions['GOOD_RSI_VALUE'][i][0]==True) and (suggestions['GOOD_MACD'][i][0]==True):
                #print(suggestions['SYMBOL'][i])
                best_for_swing_trade.append([suggestions['SYMBOL'][i],suggestions['1_MONTH_RETURN%'][i], suggestions['LAST_TRADED_PRICE_₹'][i], suggestions['20_EMA'][i]])

        #print('\n\n',pd.DataFrame(best_for_swing_trade, columns=['SYMBOL','1M_MONTH_RETURN%']).sort_values(by='1M_MONTH_RETURN%',ascending=False))
        return pd.DataFrame(best_for_swing_trade, columns=['SYMBOL','1_MONTH_RETURN%','PRICE','LIMIT_PRICE']).sort_values(by='PRICE',ascending=True).reset_index(drop=True)
        
    def LongTerm(self):
        nifty_filtered = []
        for i in range(1,len(self.nifty_100_data)):
            try:
                if (float(self.nifty_100_data['1Y_CHANGE%'][i])>50):
                    nifty_filtered.append([self.nifty_100_data['SYMBOL'][i],self.nifty_100_data['1Y_CHANGE%'][i]])
            except:
                continue

        nifty_filtered = pd.DataFrame(nifty_filtered, columns = ['SYMBOL','1_YEAR_RETURN%'])
        #print(nifty_filtered)
        #suggestions
        suggestions = []
        print('\n Please wait this might take a few seconds... \n')
        for i in range(len(nifty_filtered)):
            try:
                LTP = round(float(yf.Ticker(nifty_filtered['SYMBOL'][i]+'.NS').history(period='1d')['Close'][-1]),ndigits=2)
                suggestions.append([nifty_filtered['SYMBOL'][i],nifty_filtered['1_YEAR_RETURN%'][i],[Technicals(nifty_filtered['SYMBOL'][i] + '.NS').RSI()],[Technicals(nifty_filtered['SYMBOL'][i] + '.NS').EMA(timeframe=50)<LTP],[Technicals(nifty_filtered['SYMBOL'][i] + '.NS').EMA(timeframe=200)<LTP],[Technicals(nifty_filtered['SYMBOL'][i] + '.NS').MACD()],LTP,round(Technicals(nifty_filtered['SYMBOL'][i] + '.NS').EMA(timeframe=20, interval= "60m"),ndigits=-1)])
            except IndexError:
                print('Data unavailable for ' + nifty_filtered['SYMBOL'][i])
                continue

        suggestions = pd.DataFrame(suggestions, columns = ['SYMBOL','1_YEAR_RETURN%','GOOD_RSI_VALUE','LTP_ABOVE_50_EMA','LTP_ABOVE_200_EMA','GOOD_MACD','LAST_TRADED_PRICE_₹','20_EMA'])
        #print(suggestions)
        print('\nBest stocks to invest in, at the moment for Long Term...')
        best_for_long_term = []
        for i in range(len(suggestions)):
            if ((suggestions['LTP_ABOVE_50_EMA'][i][0]==True) or (suggestions['LTP_ABOVE_200_EMA'][i][0]==True)) and (suggestions['GOOD_RSI_VALUE'][i][0]==True) and (suggestions['GOOD_MACD'][i][0]==True):
                #print(suggestions['SYMBOL'][i])
                best_for_long_term.append([suggestions['SYMBOL'][i],suggestions['1_YEAR_RETURN%'][i],suggestions['LAST_TRADED_PRICE_₹'][i], suggestions['20_EMA'][i]])

        #print('\n\n',pd.DataFrame(best_for_swing_trade, columns=['SYMBOL','1M_MONTH_RETURN%']).sort_values(by='1M_MONTH_RETURN%',ascending=False))
        return pd.DataFrame(best_for_long_term, columns=['SYMBOL','1_YEAR_RETURN%','PRICE','LIMIT_PRICE']).sort_values(by='PRICE',ascending=True).reset_index(drop=True)


#%%

#obj = LSTMPrediction('PNB.NS')
#df, dictionary = obj.fetchFromYahoo()


#obj2 = Technicals('TATAMOTORS.NS')
#EMA = obj2.EMA(20)
#obj2.MACD()
#df_new, RSI = obj2.RSI()
#d =obj2.BollingerBands()
#df = pd.read_csv('../MW-NIFTY-100-02-Jun-2021 (4).csv')
#sla = StockListAnalysis()
#sla.SwingTrade()
#%%
#s = df.sort_values("PRICE").reset_index(drop=True)
#df
#%%

#a,b = sla.AnalyzeNiftySectors()
#sla.LongTerm()
#best_suggestions = sla.SwingTrade()

# %%
#yf.Ticker('PEL.NS').history(period='1d')['Close'][-1]
# %%

# %%
import yfinance as yf
tech = Technicals('WIPRO.NS')
EMA_50 = tech.EMA(timeframe=50,plot=True,interval='1h')
print('Moving average of WIPRO.NS: ' + str(EMA_50))




# %%
import yfinance as yf
tech = Technicals('WIPRO.NS')
RSI = tech.MACD(plot=True)


# if RSI > 70:
#     print('RSI of WIPRO.NS is in the overbought zone')
# elif RSI < 30:
#     print('RSI of WIPRO.NS is in the oversold zone')
# else:
#     print('RSI of WIPRO.NS is neither in oversold nor in overbought zone')




# %%
