
'''
 * Copyright (C) Mohand Goudjil & Victor de Fays - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Mohand Goudjil <mohand.goudjil@outlook.com> & Victor de Fays <victor.defays@gmail.com>, May 2018
'''

import talib as tb
import numpy as np
import pandas as pd
import uuid
import random
import matplotlib.pyplot as plt
import sys

class Portfolio:

    def __init__(self, initial_investment):

        self.initial_investment = initial_investment
        self.bank_capital = initial_investment
        self.invested_capital = 0
        self.borrowed_capital = 0
        self.print_result = False
        self.limit_bank_capital = 30
        #self.rf = 0.0296 #10 Year Treasury Rate - June https://ycharts.com/indicators/10_year_treasury_rate
        #self.rm = 0.08 # http://thecrix.de/#page-top 

        #param optimization
        self.weights_indicators = 0

        self.fees_rate = 0.005
        self.open_positions = pd.DataFrame({'timestamp': [], 'exchange': [], 'coin': [], 'price': [],'stop_loss_price': [],'stop_profit_price': [],'quantity': [],\
        'last_timestamp': [],'last_price': [],'fees': [],'brut_profit': [],'netto_profit':[],'netto_profit_percent':[]})

        self.historical_positions = self.open_positions.copy()
        self.signals_all = pd.DataFrame({'buy': [],'sell': []}) #Buy/Sell signal 
        self.signals_taken = pd.DataFrame({'buy': [],'sell': []}) #Buy/Sell signal 


    def add_capital(self,value_to_add):
        '''
        You can add capital to your bank account
        '''
        self.bank_capital  = self.bank_capital + value_to_add
        if self.bank_capital < 0 :
            self.bank_capital  = 0

    def add_position(self,add_position_dict): 
        '''
        You can add a position with this function. You need to input 
        a dict with 'timestamp','exchange','coin','price','quantity'
        '''

        position_cost = add_position_dict['price'] * add_position_dict['quantity']

        if (add_position_dict['quantity'] != 0) and (abs(position_cost) < self.bank_capital and self.bank_capital > self.limit_bank_capital): #Need enough money  #pas de leverage possible

            uid = str(uuid.uuid4())

            for name in add_position_dict: 
                self.open_positions.at[uid,name] = add_position_dict[name] #Fill the dict with 'timestamp','exchange','coin','price','quantity','stop_loss' WITH UNIQUE UUID
            
            if position_cost >= 0: #Long position
                
                self.bank_capital = self.bank_capital - (position_cost + self.fees_rate*position_cost)
                self.invested_capital = self.invested_capital + position_cost  
                self.open_positions.at[uid,'fees'] = self.fees_rate*position_cost #add fees
                self.signals_taken.at[add_position_dict['timestamp'],'buy'] = 1 #buy signal

                if self.print_result:
                    print("Long position of %s successfully opened"%(position_cost))
                    print("bank capital %s and position cost %s"%(self.bank_capital,position_cost))

            else: #Short position
                
                self.borrowed_capital = self.borrowed_capital + position_cost*(-1)
                self.open_positions.at[uid,'fees'] = self.fees_rate*position_cost*(-1) #add fees
                self.signals_taken.at[add_position_dict['timestamp'],'sell'] = -1 #sell signal


                if self.print_result:
                    print("Successful opening for a short position of %s USD"%(position_cost))    
                    print("bank capital %s and position cost %s"%(self.bank_capital,position_cost))
            
        else:
            if self.print_result:
                print("Unsuccessful opening of a position")   


    def update_open_positions(self, price_dict): # 'last_timestamp': , exchange': , 'coin': , 'last_price': 
        '''
        To uodate the open positions you need to input a dict with
        'last_timestamp': , exchange': , 'coin': , 'last_price': 
        '''
        #for  i in range(len(price_dict['exchange'])): 

        ## price to update
        self.open_positions.loc[ (self.open_positions['exchange'] == price_dict['exchange']) & (self.open_positions['coin'] == price_dict['coin']),'last_price'] \
             = price_dict['last_price'] #last_price

        self.open_positions.loc[ (self.open_positions['exchange'] == price_dict['exchange']) & (self.open_positions['coin'] == price_dict['coin']),'last_timestamp'] \
             = price_dict['last_timestamp'] #last_timestamp

        #brut_profit
        self.open_positions['brut_profit'] =  (self.open_positions['last_price'] - self.open_positions['price'] )  * self.open_positions['quantity'] 

        #netto_profit
        self.open_positions['netto_profit'] = self.open_positions['brut_profit'] - self.open_positions['fees']

        #netto_profit_percent
        self.open_positions['netto_profit_percent'] = (self.open_positions['brut_profit'] - self.open_positions['netto_profit'])/self.open_positions['brut_profit']

        #print("Positions succesfully updated")


    def close_open_position(self,price_dict,uuid_val): #need last price update to close position
        '''
        To close the open positions you need to input 
        a price dict => 'last_timestamp': , exchange': , 'coin': , 'last_price': 
        uuid_val => unique id of the position
        '''
        self.update_open_positions(price_dict) #Update price in order to sell with the right one         
        self.historical_positions.at[str(uuid_val),:] =  self.open_positions.loc[uuid_val].values #Copy position in historical dataframe

        if self.open_positions.loc[uuid_val]['quantity'] > 0 :
            self.bank_capital = self.bank_capital + (self.open_positions.loc[uuid_val]['quantity'] * self.open_positions.loc[uuid_val]['last_price'])
            #print('long position closed')
        else:
            self.bank_capital = self.bank_capital + self.open_positions.loc[uuid_val]['netto_profit']
            #print('short position closed')

        self.open_positions.drop(uuid_val, inplace=True) #Delete position from open positions

        if self.print_result:
            print("Position %s succesfully closed"%(uuid_val))


    def get_short_positions(self):
        return self.open_positions[self.open_positions['quantity']<0]

    def get_long_positions(self):
        return self.open_positions[self.open_positions['quantity']>0]

    def stop_loss(self,low_price_dict):

        #Close all position that declenched the stop_loss during the period thanks to OHLC data
        #{'last_timestamp': last_timestamp, 'exchange': 1, 'coin': 1, 'last_price': low_price}

        long_positions_stop_loss_activated = self.open_positions[ (low_price_dict['last_price'] <= self.open_positions['stop_loss_price']) & (self.open_positions['quantity']>0) ]
        short_positions_stop_loss_activated = self.open_positions[ (low_price_dict['last_price'] <= self.open_positions['stop_loss_price']) & (self.open_positions['quantity']<0) ]

        for index_uuid, position in long_positions_stop_loss_activated.iterrows():
            position['last_price']  = position['stop_loss_price']
            self.close_open_position(position,index_uuid)

        for index_uuid, position_short in short_positions_stop_loss_activated.iterrows():
            position_short['last_price']  = position_short['stop_loss_price']
            self.close_open_position(position_short,index_uuid)

    def close_all_position_same_price(self,close_price_dict):
        for index_uuid, _ in self.open_positions.iterrows(): 
            self.close_open_position(close_price_dict,index_uuid)
    
            sys.exit("Erreur Bank capital doesn't correspond to cash flows")


## Others functions

def buy_sell_portoflio(portfolio,signal,qt_to_buy,close_price_dict,low_price_dict,stop_loss_pct):

    portfolio.stop_loss(low_price_dict) #Close positions that overtake stop_loss 

    long_positions = portfolio.get_long_positions()
    short_positions = portfolio.get_short_positions()
    
    if signal != 0 : 

        if signal == 1: #close all  short position
            for index_uuid, _ in short_positions.iterrows(): 
                portfolio.close_open_position(close_price_dict,index_uuid)
                
        elif  signal == -1 : #close all long position
            for index_uuid, _ in long_positions.iterrows():
                portfolio.close_open_position(close_price_dict,index_uuid)

        if signal == 1 : #or signal == -1 :
            # Buy long/short position
            add_position_dict = close_price_dict.copy()
            add_position_dict['price'] = add_position_dict.pop('last_price')
            add_position_dict['timestamp'] = add_position_dict.pop('last_timestamp')

            # Add info to add position dict
            add_position_dict['quantity'] = qt_to_buy*signal
            add_position_dict['stop_loss_price'] = add_position_dict['price'] - (add_position_dict['price'] * stop_loss_pct * signal) # Stop Loss at specific position

            portfolio.add_position(add_position_dict)

    portfolio.update_open_positions(close_price_dict)


def launch_backtesting(df_price,df_output,portfolio,stop_loss_val,weight_p): #,start_row,end_row) : #,weights_array):#optimization_parameters =0):

  for last_timestamp, df_row_price in df_price.iterrows(): #df_price[start_row:end_row].iterrows():

    ## Price data
    close_price = df_row_price['Close']
    close_price = round(close_price, 4)
    low_price = df_row_price['Low']
    low_price = round(close_price, 4)

    signal = df_output.loc[last_timestamp][0]
    
    if signal == 1 :
        portfolio.signals_all.at[last_timestamp,'buy'] = 1
    elif signal == -1:
        portfolio.signals_all.at[last_timestamp,'sell'] = -1
    
    ## Qt_to_buy
    qt_to_buy = (portfolio.bank_capital / close_price) / weight_p #random.randint(10, 100) 
    qt_to_buy = round(qt_to_buy, 3)

    ## Price dict
    close_price_dict = {'last_timestamp': last_timestamp, 'exchange': 1, 'coin': 1, 'last_price': close_price}
    low_price_dict = {'last_timestamp': last_timestamp, 'exchange': 1, 'coin': 1, 'last_price': low_price}
    stop_loss_pct = stop_loss_val

    buy_sell_portoflio(portfolio,signal,qt_to_buy,close_price_dict,low_price_dict,stop_loss_pct*(0.1))
    
  return portfolio

  #Technical_indicators

def trading_logic(df_price,df_previous_price,weights_array):

    # param_matrix => Weight
    signal = 0 # hold

    ### 0. ADX

    if (df_previous_price['di_plus'] < df_previous_price['di_minus']) and (df_price['di_plus'] > df_price['di_minus']) and (df_price['adx'] > 20):
        adx = 1
    elif (df_previous_price['di_plus'] > df_previous_price['di_minus']) and (df_price['di_plus'] < df_price['di_minus']) and (df_price['adx'] > 20):
        adx = -1
    else:
        adx = 0

    ### 1. Bollinger Bands

    if df_previous_price['Close'] < df_previous_price['lowerband'] and df_price['Close'] > df_price['lowerband']:
        bbsig = 1
    elif df_previous_price['Close'] < df_previous_price['upperband'] and df_price['Close'] > df_price['upperband']:
        bbsig = -1
    else:
        bbsig = 0

    ### 2. Commodity-Channel-Index

    if df_previous_price['cci']< -100 and df_price['cci'] > -100:
        ccisig = 1
    elif df_previous_price['cci']< 100 and df_price['cci'] > 100:
        ccisig = -1
    else:
        ccisig = 0

    ### 3. Exponential MA  5 vs 21

    if df_previous_price['ema5'] < df_previous_price['ema21'] and df_price['ema5'] > df_price['ema21']:
        emasig = 1
    elif df_previous_price['ema5'] > df_previous_price['ema21'] and df_price['ema5'] < df_price['ema21']:
        emasig = -1
    else:
        emasig = 0
    
    ### 4. Moving-Averages-Convergence-Divergence

    if df_previous_price['macd'] < df_previous_price['macdema'] and df_price['macd'] > df_price['macdema']:
        macdsig = 1
    elif df_previous_price['macd'] > df_previous_price['macdema'] and df_price['macd'] < df_price['macdema']:
        macdsig = -1
    else:
        macdsig = 0

    ### 5. Rate of change

    if df_previous_price['roc'] < -10 and df_price['roc'] > -10:
        rocsig = 1
    elif df_previous_price['roc'] < 10 and df_price['roc'] > 10:
        rocsig = -1
    else:
        rocsig = 0
    
    ### 6. RSI

    if df_previous_price['rsi'] < 30 and df_price['rsi'] > 30:
        rsisig = 1
    elif df_previous_price['rsi'] < 70 and df_price['rsi'] > 70:
        rsisig = -1
    else:
        rsisig = 0

    ### 7. WMA-and-CCI

    if df_previous_price['Close'] < df_previous_price['wma5'] and df_price['Close'] > df_price['wma5'] and df_price['cci'] < -100:
        ccismasig = 1
    elif df_previous_price['Close'] > df_previous_price['wma5'] and df_price['Close'] < df_price['wma5'] and df_price['cci'] > 100:
        ccismasig = -1
    else:
        ccismasig = 0

    ### 8. WMA-and-RSI

    if df_previous_price['Close'] < df_previous_price['wma5'] and df_price['Close'] > df_price['wma5'] and df_price['rsi'] < 30:
        rsismasig = 1
    elif df_previous_price['Close'] > df_previous_price['wma5'] and df_price['Close'] < df_price['wma5'] and df_price['rsi'] > 70:
        rsismasig = -1
    else:
        rsismasig = 0
    
    ### 9. WMA-and-STO.py

    if df_previous_price['Close'] < df_previous_price['wma5'] and df_previous_price['Close'] > df_previous_price['wma5'] and df_previous_price['slowd'] < 20:
        stosmasig = 1
    elif df_previous_price['Close'] > df_previous_price['wma5'] and df_previous_price['Close'] < df_previous_price['wma5'] and df_previous_price['slowd'] > 80:
        stosmasig = -1
    else:
        stosmasig = 0
    
    ### 10. Double Exponential MA  5 vs 21

    if df_previous_price['dema5'] < df_previous_price['dema21'] and df_price['dema5'] > df_price['dema21']:
        demasig = 1
    elif df_previous_price['dema5'] > df_previous_price['dema21'] and df_price['dema5'] < df_price['dema21']:
        demasig = -1
    else:
        demasig = 0
    
    ### 11. Triple Exponential MA  and Chaikin

    if df_previous_price['Close'] < df_previous_price['tema'] and df_price['Close'] > df_price['tema'] and df_price['chaikin'] > 0.05:
        temack = 1
    elif df_previous_price['Close'] > df_previous_price['tema'] and df_price['Close'] < df_price['tema'] and df_price['chaikin'] < -0.05:
        temack = -1
    else:
        temack = 0
    
    ### 11. Triangular Moving Average
    
    if df_previous_price['trima5'] < df_previous_price['trima30'] and df_price['trima5'] > df_price['trima30']:
        trima = 1
    elif df_previous_price['trima5'] > df_previous_price['trima30'] and df_price['trima5'] < df_price['trima30']:
        trima = -1
    else:
        trima = 0
    ### 
    
    signals_array = np.array([adx,bbsig,ccisig,emasig,macdsig,rocsig,rsisig,ccismasig,rsismasig,stosmasig,demasig,temack,trima])
    #weights_array = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])
    final_signal_value = (signals_array * weights_array).mean()

    if  final_signal_value >  0.3 :
        signal = 1
    elif final_signal_value < -0.3:
        signal = -1
    else:
        signal = 0

    return signal

def technical_indicators_panda_dataframe(df_price):
 
  open = df_price['Open']
  high = df_price['High']
  low = df_price['Low']
  close = df_price['Close']
  volume = df_price['Volume']

    ## Commoditty Channel Index
  cci= tb.CCI(high, low, close, timeperiod=20)

    ## Rate of change
  roc = tb.ROC(close, timeperiod=21)

    ## Momentum Indicators Functions

  #Aroon
  aroondown, aroonup = tb.AROON(high, low, timeperiod=14)

  #Average Directional Movement Index
  adx = tb.ADX(high, low, close, timeperiod=14)
  di_plus = tb.PLUS_DI(high, low, close, timeperiod=14)
  di_minus = tb.MINUS_DI(high, low, close, timeperiod=14)

  #Money Flow Index
  mfi = tb.MFI(high, low, close, volume, timeperiod=14)

  #Relative Strength Index
  rsi = tb.RSI(close, timeperiod=14)

  #Stochastic
  slowk, slowd = tb.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

  #STOCHF - Stochastic Fast
  fastk, fastd = tb.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)

  #Moving Average Convergence/Divergence
  macd, macdema, macdhist = tb.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    ## Overlap Studies Functions 

  #Bollinger Bands
  upperband, middleband, lowerband = tb.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)

  #Weighted Moving Average
  wma5 = tb.WMA(close, timeperiod=5)
  wma30 = tb.WMA(close, timeperiod=30)

  #Exponential Moving Average
  ema5 = tb.EMA(close, timeperiod=5)
  ema21 = tb.EMA(close, timeperiod=21)

  #Double Exponential Moving Average
  dema5 = tb.DEMA(close, timeperiod=5)
  dema21 = tb.DEMA(close, timeperiod=21)

  #Triple Exponential Moving Average 
  tema = tb.TEMA(close, timeperiod=30)

  #Triangular Moving Average
  trima5 = tb.DEMA(close, timeperiod=5)
  trima30 = tb.TRIMA(close, timeperiod=30)

    ## Volume indicators Functions 

  #AD - Chaikin A/D Line
  chaikin = tb.AD(high, low, close, volume)

  ##=> MERGE
  
  kwargs = {"cci":cci, "roc": roc, "aroondown": aroondown, "aroonup": aroonup, "adx": adx, "di_plus":di_plus, "di_minus":di_minus, "mfi": mfi, "rsi": rsi, "slowk": slowk, "slowd": slowd, "fastk": fastk, "fastd": fastd, "macd": macd, "macdema": macdema, "macdhist": macdhist,\
    "upperband": upperband , "middleband": middleband , "lowerband": lowerband, "wma5" : wma5, "wma30" : wma30,  "ema5" : ema5, "ema21" : ema21, "dema21": dema21, "dema5": dema5, "tema": tema, "trima5": trima5,"trima30": trima30,\
    "chaikin": chaikin}

  return df_price.assign(**kwargs)