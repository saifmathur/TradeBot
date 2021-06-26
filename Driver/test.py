#%%

import pandas as pd
import yfinance as yf
import sys
sys.path.append('../DLpart/')
from PredictStock import Technicals
import datetime

nifty100Columns = ['SYMBOL','OPEN','HIGH','LOW','PREVCLOSE','LTP','TODAYS_CHANGE',
                    'CHANGE_%','VOLUME','VALUE','52WH','52WL','1Y_CHANGE%','1M_CHANGE%']

niftySectorsColumns = ['INDEX','CURRENT','%CHANGE','OPEN','HIGH','LOW','PREVCLOSE','PREVDAY','1W_CLOSE','1M_CLOSE','1Y_CLOSE',
                        '52WH','52WL','1Y_CHANGE%','1M_CHANGE%']
    
nifty100 = pd.read_csv('../PreFedIndexData/MW-NIFTY-100-'+datetime.datetime.strftime(datetime.datetime.today(),"%d-%b-%Y")+'.csv', names = nifty100Columns, header=0)
niftyAllIndice = pd.read_csv(str('../PreFedIndexData/MW-All-Indices-')+datetime.datetime.strftime(datetime.datetime.today(),"%d-%b-%Y")+'.csv', names = niftySectorsColumns, header=0)

NIFTY_SECTORS = pd.DataFrame(niftyAllIndice.iloc[[13,14,15,17,18,19,20,21,22,23,24,44,45],:].values, columns=niftySectorsColumns)

NIFTY_SECTORS = NIFTY_SECTORS.drop(index=3)
NIFTY_SECTORS = NIFTY_SECTORS.reset_index(drop=True)
#SECTOR_NAME_ON_YAHOO_FINANCE = ['Banks—Regional','Auto Manufacturers','Auto']
#NIFTY_SECTORS.columns
Highest_1Y_return_sectors = []
for i in range(len(NIFTY_SECTORS)):
    if float(NIFTY_SECTORS['1Y_CHANGE%'][i]) > 50:
        #print(NIFTY_SECTORS['INDEX'][i])
        Highest_1Y_return_sectors.append(NIFTY_SECTORS['INDEX'][i])




#%%
nifty_filtered = []

#len(nifty100
for i in range(1,len(nifty100)):
    try:
        if (float(nifty100['1Y_CHANGE%'][i])>50) and (float(nifty100['1M_CHANGE%'][i])>5):
            #nifty100['1Y_CHANGE%'][i]
            #print(nifty100['SYMBOL'][i])
            nifty_filtered.append([nifty100['SYMBOL'][i],nifty100['1M_CHANGE%'][i]])
    except:
        continue

nifty_filtered = pd.DataFrame(nifty_filtered, columns = ['SYMBOL','1_MONTH_RETURN%'])



#%%
nifty_filtered.dtypes

#%%
suggestions = []
print('\n Please wait this might take a few seconds... \n')
for i in range(len(nifty_filtered)):
    #print(nifty_filtered['SYMBOL'][i],nifty_filtered['1_MONTH_RETURN%'][i],Technicals(nifty_filtered['SYMBOL'][i] + '.NS').RSI(),[Technicals(nifty_filtered['SYMBOL'][i] + '.NS').EMA(timeframe=50)>=LTP])
    #yobj = yf.Ticker(self.symbol)
    #df = yobj.history(period="1y")
    LTP = round(float(yf.Ticker(nifty_filtered['SYMBOL'][i]+'.NS').history(period='1d')['Close'][-1]),ndigits=2)
    # print(nifty_filtered['SYMBOL'][i],
    #         nifty_filtered['1_MONTH_RETURN%'][i],
    #         #[Technicals(nifty_filtered['SYMBOL'][i] + '.NS').RSI()<=55 ],
    #         [Technicals(nifty_filtered['SYMBOL'][i] + '.NS').RSI()],
    #         [Technicals(nifty_filtered['SYMBOL'][i] + '.NS').EMA(timeframe=50)<LTP],
    #         [Technicals(nifty_filtered['SYMBOL'][i] + '.NS').MACD()])
    #suggestions.append([nifty_filtered['SYMBOL'][i],nifty_filtered['1_MONTH_RETURN%'][i],[Technicals(nifty_filtered['SYMBOL'][i] + '.NS').RSI()<=60],[Technicals(nifty_filtered['SYMBOL'][i] + '.NS').EMA(timeframe=50)<LTP],[Technicals(nifty_filtered['SYMBOL'][i] + '.NS').MACD()]])
    suggestions.append([nifty_filtered['SYMBOL'][i],nifty_filtered['1_MONTH_RETURN%'][i],[Technicals(nifty_filtered['SYMBOL'][i] + '.NS').RSI()],[Technicals(nifty_filtered['SYMBOL'][i] + '.NS').EMA(timeframe=50)<LTP],[Technicals(nifty_filtered['SYMBOL'][i] + '.NS').MACD()]])

suggestions = pd.DataFrame(suggestions, columns = ['SYMBOL','1_MONTH_RETURN%','GOOD_RSI_VALUE','LTP_ABOVE_50_EMA','GOOD_MACD'])
print('\n', suggestions)

#%%
# industries = []
# for i in range(len(nifty_filtered)):
#     #print(nifty_filtered['SYMBOL'][i], yf.Ticker(nifty_filtered['SYMBOL'][i]+'.NS').get_info()['industry'])
#     industries.append(yf.Ticker(nifty_filtered['SYMBOL'][i]+'.NS').get_info())

# print('Done')

# %%
#print(str(yf.Ticker('TATAMOTORS.NS').get_info()['industry']).find('Auto'))
# if str(yf.Ticker('TATAMOTORS.NS').get_info()['industry']).find('Auto') != -1:
#     print('NIFTY AUTO')

# %%
# for i in nifty_filtered['SYMBOL']:
#     print(yf.Ticker(i + '.NS').get_info()['industry'])
# %%
#round(float(yf.Ticker('TATAMOTORS.NS').history(period='1d')['Close'][-1]),ndigits=2)

# %%

# %%
# Technicals('NMDC.NS').RSI()
# # %%
# suggestions.columns
# suggestions.dtypes

#%%
suggestions['1_MONTH_RETURN%'].mean()
#%%
print('\nBest stocks to invest in, at the moment for swing trading...')
best_for_swing_trade = []
for i in range(len(suggestions)):
    if (suggestions['LTP_ABOVE_50_EMA'][i][0]==True) and (suggestions['GOOD_RSI_VALUE'][i][0]==True) or  (suggestions['GOOD_MACD'][i][0]==True):
        print(suggestions['SYMBOL'][i])
        best_for_swing_trade.append([suggestions['SYMBOL'][i],suggestions['1_MONTH_RETURN%'][i]])

print('\n\n',pd.DataFrame(best_for_swing_trade, columns=['SYMBOL','1M_MONTH_RETURN%']).sort_values(by='1M_MONTH_RETURN%',ascending=False))
# %%
import datetime

print(datetime.datetime.strftime(datetime.datetime.today(),"%d-%b-%Y"))
# %%
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
print('2 year historical data works best')
import yfinance as yf
df = yf.Ticker('^NSEI').history('2y')
df = df.iloc[:,0:4]
df.tail()

df.index

df_train = df[df.index < '2021-05-07']
df_test = df[df.index > '2021-05-07']


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import accuracy_score
regressor = LinearRegression()


sc = MinMaxScaler()
x_train = df_train.iloc[:,0:3]
y_train = df_train.iloc[:,-1]

x_test = df_test.iloc[:,0:3]
y_test = df_test.iloc[:,-1]

linear_model = regressor.fit(x_train,y_train)
prediction = regressor.predict(x_test)

score = regressor.score(x_test,y_test)
print(round(score*100,ndigits=3),'%')

# %%
regressor.predict([[15725.099609375,15773.4501953125,15678.099609375]])
# %%
def printHello():
    return 'Hello'
def what(string = printHello()):
    print(string)

# %%
what()
# %%
