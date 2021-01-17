#%%
from usingYahoo import fetchFromYahoo, get_train_test_dataset, create_dataset
import yfinance as yf
import matplotlib.pyplot as plt

print('\nNifty \n' , yf.Ticker("^NSEI").history(period='max')[::-1])
plt.plot(yf.Ticker("^NSEI").history(period='max')['Close'])
plt.title('NIFTY 50')
plt.show()

print('\nSensex \n' , yf.Ticker("^BSESN").history(period='max')[::-1])
plt.plot(yf.Ticker("^BSESN").history(period='max')['Close'])
plt.title('SENSEX')
plt.show()

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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_pred = model.predict(xtrain)
train_pred_inverse = scaler.inverse_transform(train_pred)


import matplotlib.pyplot as plt
plt.plot(train_pred)
plt.show()




# %%
