#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from sklearn.model_selection import train_test_split
import datetime as dt

yobj = yf.Ticker('TATAMOTORS.NS')
df = yobj.history(period="1y")
df.head()
df = df.drop(['Stock Splits','Dividends'],axis=1)
#df.index =  pd.to_datetime(df.index)
#%%

#df1 = df[df.index <= '2021-04-19']

train_start = dt.datetime(2020,4,22)
train_end = dt.datetime(2021,1,21)

train = df['Close'][df.index <= train_end]
test = df['Close'][df.index >= train_end]

# %%
sc = MinMaxScaler(feature_range=(0,1))
train_scaled_data = sc.fit_transform(train.values.reshape(-1,1))
test_scaled_data = sc.fit_transform(test.values.reshape(-1,1))
# %%
look_back = 60

xtrain=[]
ytrain=[]

for x in range(look_back, len(train_scaled_data)):
    xtrain.append(train_scaled_data[x-look_back:x,0])
    ytrain.append(train_scaled_data[x,0])

xtrain, ytrain = np.array(xtrain), np.array(ytrain)
xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1],1))

# %%
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(xtrain.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain,ytrain,epochs=25, batch_size=32)

# %%
''' Testing '''
import datetime as dt
xtest = []
for x in range(look_back, len(test_scaled_data)):
    xtest.append(test_scaled_data[x-look_back:x,0])

#%%
xtest = np.array(xtest)
xtest = np.reshape(xtest, (xtest.shape[0],xtest.shape[1],1))
#%%
test_scaled_data = np.array(test_scaled_data)
test_scaled_data = np.reshape(test_scaled_data, (test_scaled_data.shape[0],test_scaled_data.shape[1],1))
#%%
predictions = model.predict(test_scaled_data)
predictions_inverse = sc.inverse_transform(predictions)

# %%

# %%
