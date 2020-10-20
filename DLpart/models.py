# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.models import Sequential
from FetchData import fetchWeekly, fetchDaily, fetchIntraday
from key import KEY

df = fetchDaily('BSE:SBIN',KEY)

# %%
