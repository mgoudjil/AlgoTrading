
#Libraries

from crypto_trading import *
from pandas import read_csv
import matplotlib.pyplot as plt
import sys 
import multiprocessing as mp
import numpy as np
import itertools
import pickle
from classifier_all import get_data

#Genetic 
#from GA_optimization import *

coin_name = "eth"
df_price = read_csv('./backtesting/%s_price.csv'%(coin_name), header=0, index_col=0)
df_output = read_csv('./backtesting/%s_output.csv'%(coin_name), header=0, index_col=0) 
start_row = 4970
end_row = 5000

# Start
start_investment = 100000
port = launch_backtesting(df_price,df_output,Portfolio(start_investment),0.01,100)
port.historical_positions['netto_profit'].sum()


##### PLOT 

portfolio1 = port
all_signal = True

df_price.index = pd.to_datetime(df_price.index,unit='s')
portfolio1.signals_all.index = pd.to_datetime(portfolio1.signals_all.index,unit='s')
portfolio1.signals_taken.index = pd.to_datetime(portfolio1.signals_taken.index,unit='s')

if all_signal:
  #All signals
  buy_signal = portfolio1.signals_all[portfolio1.signals_all['buy'] == 1]['buy']
  sell_signal = portfolio1.signals_all[portfolio1.signals_all['sell'] == -1]['sell']
else:
  #Taken signals
  buy_signal = portfolio1.signals_taken[portfolio1.signals_all['buy'] == 1]['buy']
  sell_signal = portfolio1.signals_taken[portfolio1.signals_all['sell'] == -1]['sell']

plt.plot(buy_signal.index, df_price.ix[buy_signal.index]['Close'], '^', markersize=10, color='g',label='Buy')
plt.plot(sell_signal.index, df_price.ix[sell_signal.index]['Close'], 'v', markersize=10, color='r',label='Sell')

  #Price
plt.plot(df_price[24:48].index, df_price[24:48]['Close'], label='Bitcoin')

plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend(loc=0)
plt.show()

##### END PLOT 



