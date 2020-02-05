
import pandas as pd
import numpy as np
from pandas import read_csv , DataFrame, concat
from pandas import Series

from sklearn.datasets import load_digits

from helper_module import *
#from sklearn.decomposition import PCA

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn.metrics import accuracy_score

from statistics import mean
import matplotlib.pyplot as plt
import sys
from random import randint
from sklearn.model_selection import RandomizedSearchCV
import multiprocessing as mp
import pickle
from sklearn.externals import joblib
from hpsklearn import HyperoptEstimator, svc
from hpsklearn import HyperoptEstimator, any_classifier
from hyperopt import tpe
from sklearn.preprocessing import MinMaxScaler
    ############### A. Prepare Data ##############

def get_data(lagged_window,select_feature_on,diff_on,window_var_on,type_indicators,coin_val,file_source_date):
     
    if coin_val == "btc":
        df = read_csv('./Data/%s/btc_merged.csv'%(file_source_date), header=0, index_col=0)
    else:
        df = read_csv('./Data/%s/eth_merged.csv'%(file_source_date), header=0, index_col=0)


        # 1. Generate Output y BUY/SELL for period t+1
    df['Close1'] = df['Close'].pct_change().shift(-1)
    threeshold_signal = 0.00

    df.loc[ df['Close1'] >= threeshold_signal , 'output_y'] = 1 #BUY
    df.loc[ df['Close1'] < threeshold_signal, 'output_y'] = -1 #SELL

        # 2. Configure outputs/inputs      
    y_output = pd.DataFrame({'output_y': df['output_y'].values },index=df.index).fillna(1) #Y ouput
    y_output = np.ravel(y_output)
    x_input = df.drop(['Close1','output_y'],axis=1).fillna(0)      

        # 3. Choose variables columns
    if coin_val == "btc" :
        if type_indicators == 1:
            x_input = x_input.iloc[:,:31] #Technical BTC
        elif type_indicators == 2:
            x_input = x_input.iloc[:,31:35] #Social BTC
        elif type_indicators == 3:
            x_input = x_input.iloc[:,35:] #Fundamental BTC
        elif type_indicators == 4:
            x_input = x_input.iloc[:,:35] #HYBRID BTC

    else:
        if type_indicators == 1:
            x_input = x_input.iloc[:,:33] #Technical ETH
        elif type_indicators == 2:
            x_input = x_input.iloc[:,33:37] #Social ETH
        elif type_indicators == 3:
            x_input = x_input.iloc[:,37:] #Fundamental ETH
        elif type_indicators == 4:
            x_input = x_input.iloc[:,:35] #Hybrid ETH


    ############### B. Pre_process ##############

        #1. Features selection
    if len(x_input) != len(y_output):
        sys.exit("Fatal Error: len X/Y different")

    if window_var_on == 1:
        window_var = x_input.rolling(window=randint(2,10))
        x_input = concat([window_var.min(), window_var.mean(), window_var.max(),window_var.std(), x_input], axis=1)

    if select_feature_on == 1:
        forest = ExtraTreesClassifier()  # Build a forest  
        forest = forest.fit(x_input,y_output)

        model = SelectFromModel(forest, prefit=True) # Compute the feature importances
        x_input = model.transform(x_input)
    else:
        x_input = x_input.values

        #2. Introduce lag variables
    x_input = pd.DataFrame(data = x_input ,index=df.index).fillna(1) #Y ouput

    if diff_on == 1:
        x_input = x_input.diff()
    else:
        x_input = x_input.diff()

    x_input = lagged_dataframe(x_input,lagged_window)


        #3. Prepare X/Y 
    x_input = x_input.values

    if len(x_input) != len(y_output):
        sys.exit("Fatal Error: len X/Y different")

    x_input = x_input[window_start:-window_end,]
    y_output = y_output[window_start:-window_end,]

    return (x_input, y_output)


    ############### C. ML Algorithms ##############

from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

def ml_algo(x_input,y_output) :

    #1. Neural Network - MLPClassifier

    if  choosen_model == "Multi-layer perceptron":

        param_search = [ {'activation': ['logistic']} ]    #   param_search = [ {'activation': ['logistic','relu'], 'learning_rate': ['constant','adaptive'], 'solver': ['adam','lbfgs']} ]    

        model = MLPClassifier(solver='adam',learning_rate = 'constant', hidden_layer_sizes=(10, 2), random_state=None ,max_iter= 4,alpha=0.00001,shuffle=False) 

    
   
    ############### D. MODEL VALIDATION ##############

    n_train_hours = int(len(x_input)*0.8)
    X_train,X_test = x_input[:n_train_hours, :], x_input[n_train_hours:, :]
    y_train, y_test = y_output[:n_train_hours], y_output[n_train_hours:]

    # Variables standardization
    if standardization == 1:
        (X_train, X_test) = standardize_me(X_train,X_test,standardization) #robust #min #bina

    #Time series Forward x Cross_validation 
    if choosen_model == "Multi-layer perceptron": 
        print("ok")
        nested_cross_v = TimeSeriesSplit(n_splits=cross_validation_split).split(X_train)
        gsearch = GridSearchCV(estimator=model, cv=nested_cross_v, param_grid=param_search)
        gsearch.fit(X_train, y_train)
        predicted_y = gsearch.predict(X_test)

    else:
	#2. SVM
        gsearch = HyperoptEstimator( classifier=svc('mySVC'), algo=tpe.suggest,max_evals=4 ,trial_timeout=50)
        gsearch.fit( X_train, y_train )
        predicted_y = gsearch.predict( X_test )


    confusion_matrix_final = confusion_matrix(y_test, predicted_y,labels=[1,-1])

    #score_final = f1_score(y_test, predicted_y, average='binary')  
    score_final = accuracy_score(y_test, predicted_y)
    fsdsdsds = pd.DataFrame(data = predicted_y) #Y ouput
    fsdsdsds.to_csv("backsjds_final.csv")

    return (gsearch,score_final,confusion_matrix_final)



############ STart
############### 0. Parameters ##############

### Variables
def testttt(): 
    hyper_opt_max_eval = 10
    hyperopt_trial = 60

    cross_validation_split = 10 # x fold cross-validation
    window_start = 200
    window_end= 10

    #
    best_score = [-1000]
    type_val = 4

    #Coin
    coin_val = "btc"
    file_source_date = "Hourly"

    lagged_window = 3
    select_feature_on = 1
    diff_on = 0
    window_var_on = 0
    standardization = 1

    #Choose model

    random_check = None

    choosen_model = "Support Vector Machine"
    choosen_model = "Multi-layer perceptron"

    for i in [0]:
        (x_input,y_output) = get_data(lagged_window,select_feature_on ,diff_on, window_var_on,type_val,coin_val,file_source_date)
        (last_model,score_final,confusion_matrix_final) = ml_algo(x_input, y_output)

        print(score_final)
        print(confusion_matrix_final)

