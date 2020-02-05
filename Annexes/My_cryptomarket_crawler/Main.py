#pip.main(['install', 'coinmarketcap'])
#pip.main(['install', 'numpy'])
#pip.main(['install', 'scrapy'])
##pip.main(['install', 'pandas'])
#pip.main(['install', 'beautifulsoup5'])

#import json
#import pip

import numpy as np
import pandas as pd
import pickle

from datetime import datetime
from dateutil.parser import parse

from GetCryptoWebCrap import  get_table_crypto
from coinmarketcap import Market

#GetCryptoNames
coinmarketcap = Market()
coin_data = coinmarketcap.ticker(limit=0)

nbr_crypto = len(coin_data)
crypto_names = np.array(['                                              ' for _ in range(nbr_crypto)])
crypto_id = np.array(['                                              ' for _ in range(nbr_crypto)])

for i in range( 1,nbr_crypto):
    crypto_names[i] = coin_data[i-1]["id"].replace(' ', '-').lower()

crypto_dates = get_table_crypto(crypto_names[1],"20130428","20181231").columns.values

#Dataframes

df_open = pd.DataFrame(index=crypto_dates, columns=crypto_names)
df_high = pd.DataFrame(index=crypto_dates, columns=crypto_names)
df_low  = pd.DataFrame(index=crypto_dates, columns=crypto_names)
df_close = pd.DataFrame(index=crypto_dates, columns=crypto_names)
df_volume = pd.DataFrame(index=crypto_dates, columns=crypto_names)
df_marketcap = pd.DataFrame(index=crypto_dates, columns=crypto_names)


for i in range(1,1365):
    df_get = get_table_crypto(crypto_names[i],"20130428","20181231")
    crypto_dates_get = df_get.columns.values

    for item in crypto_dates_get:
        df_open.xs(item)[crypto_names[i]]= df_get.xs("Open")[item]
        df_high.xs(item)[crypto_names[i]] = df_get.xs("High")[item]
        df_low.xs(item)[crypto_names[i]] = df_get.xs("Low")[item]
        df_close.xs(item)[crypto_names[i]] = df_get.xs("Close")[item]
        df_volume.xs(item)[crypto_names[i]] = df_get.xs("Volume")[item]
        df_marketcap.xs(item)[crypto_names[i]] = df_get.xs("MarketCap")[item]

    df_get.to_pickle("FSave_get")
    df_open.to_pickle("FSave_open")
    df_high.to_pickle("FSave_high")
    df_low.to_pickle("FSave_low")
    df_close.to_pickle("FSave_close")
    df_volume.to_pickle("FSave_volume")
    df_marketcap.to_pickle("FSave_marketcap")

#df_open.to_pickle("testtttds")
#df = pd.read_pickle('testtttds')

#print(my_array2)



