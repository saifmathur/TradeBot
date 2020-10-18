# %%
import pandas as pd
from nsetools import Nse

nse = Nse()
infy = nse.get_quote('infy')
# %%
def fetch(symbol):
    