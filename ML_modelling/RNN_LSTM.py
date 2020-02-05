


# RNN LSTM MODEL 

from __future__ import print_function

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.ndimage.interpolation import shift

from math import sqrt, floor
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform



def data():

    df = pd.read_csv('./Data/Hourly/btc_merged.csv', header=0, index_col=0)
    windows_start = 100
        # 1. Generate Output y BUY/SELL for period t+1
    df['Close1'] = df['Close'].pct_change().shift(-1)
    threeshold_signal = 0.0

    df.loc[ df['Close1'] >= threeshold_signal , 'output_y'] = 1 #BUY
    df.loc[ df['Close1'] < threeshold_signal, 'output_y'] = 0 #SELL

        # 2. Configure outputs/inputs      
    y_output = pd.DataFrame({'output_y': df['output_y'].values },index=df.index).fillna(1) #Y ouput
    y_output = np.ravel(y_output)
    x_input = df.drop(['Close1','output_y'],axis=1).fillna(0)


    y_output = df['output_y'].as_matrix()
    x_input = x_input.as_matrix()

    n_train_hours = int(len(x_input)*0.8)
    x_train,x_test = x_input[:n_train_hours, :], x_input[n_train_hours:, :]
    y_train, y_test = y_output[:n_train_hours], y_output[n_train_hours:]

    # Define the scaler 
    #scaler = StandardScaler().fit(x_train)
    scaler = MinMaxScaler().fit(x_train)

    # Scale the train set
    x_train = scaler.transform(x_train)

    # Scale the test set
    x_test = scaler.transform(x_test)
        
    # Set random seed
    #np.random.seed(7)

    print("--- shape report ---")
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("x_test: ", x_test.shape)

    # split the training for validation
    rate = 1.0 
    train_sample_size = floor(x_train.shape[0]*rate)
    # commented out the validation
    #x_valid = np.copy(x_train[train_sample_size:,:])
    #y_valid = np.copy(y_train[train_sample_size:])
    x_train = x_train[windows_start:train_sample_size,:]
    y_train = y_train[windows_start:train_sample_size]

    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    #x_valid = x_valid.reshape((x_valid.shape[0], 1, x_valid.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    print("-- network input --")
    print("X_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    # print("X_valid: ", x_valid.shape)
    # print("y_valid: ", y_valid.shape)
    print("X_test: ", x_test.shape)

    return x_train, y_train, x_test, y_test

# design network

def create_model(x_train, y_train, x_test, y_test):

    model = Sequential()
    #model.add(LSTM(100, input_shape=(x_train.shape[1], x_train.shape[2])))
    
    model.add(LSTM({{choice([20, 100, 150,200,300,500])}},return_sequences=True, stateful=False, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(50)) 
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(1))
    model.add(Activation('tanh')) 

    model.compile(loss='binary_crossentropy', optimizer={{choice(['adam','sgd'])}},metrics=['accuracy'])

    # fit network
    #history = model.fit(x_train, y_train, , batch_size=, validation_data=(x_valid, y_valid), verbose=0, shuffle=False)
    early_stopping = EarlyStopping(monitor='binary_accuracy', patience=10)
    checkpointer = ModelCheckpoint(filepath='keras_weights2.hdf5',
                                   verbose=1,
                                   save_best_only=True)
    model.fit(x_train, y_train,
              batch_size={{choice([50,150,300,400,500,1000])}},
              epochs={{choice([10,20,60,100,300,500])}},
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[early_stopping, checkpointer])

              
    #history_global.append(history)

    score, acc = model.evaluate(x_test, y_test, verbose=0)
    with open('/trainHistoryDict%s'%(score), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    history_global = []
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=200,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    best_model.save('my_model_final_performance.h5')
    print("Best performing model chosen hyper-parameters:")
    print(best_run)


