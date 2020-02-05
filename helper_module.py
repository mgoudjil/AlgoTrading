import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler , MinMaxScaler, Binarizer, RobustScaler


def positive_tobinary(x):

    if x > 0:
        value = 1
    else:
        value = 0

    return value

def suffixed_columns(df, suffix):
    return ["{}{}".format(column, suffix) for column in df.columns]

def constants(df):
    return pd.DataFrame(np.ones_like(df), index=df.index, columns=suffixed_columns(df, "_Int"))

def lag(df, n):
    new_df = df.shift(n)
    new_df.columns = suffixed_columns(df, "_Lag{:02d}".format(n))
    return new_df

def lagged_dataframe(df, lags=1):
    return pd.concat([df, constants(df)] + [lag(df, i) for i in range(1, lags + 1)], axis=1)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def standardize_me(train_X,test_X,standardization):

    if standardization == 1 :
        #scaler = StandardScaler() 
        #scaler = MinMaxScaler()
        scaler = RobustScaler()

            #Fit only on training data
        scaler.fit(train_X)  
        train_X = scaler.transform(train_X)
            #Apply same transformation to test data
        test_X = scaler.transform(test_X) 



    elif standardization == 2 :
        #Binary
        scaler = MinMaxScaler()
        scaler.fit(train_X)  
        train_X = scaler.transform(train_X)
            #Apply same transformation to test data
        test_X = scaler.transform(test_X)  
    
    elif standardization == 3 :
        #Binary
        scaler = Binarizer() 
        scaler.fit(train_X)  
        train_X = scaler.transform(train_X)
            #Apply same transformation to test data
        test_X = scaler.transform(test_X)  

    elif standardization == 4 :
        #Binary
        scaler = StandardScaler() 
        scaler.fit(train_X)  
        train_X = scaler.transform(train_X)
            #Apply same transformation to test data
        test_X = scaler.transform(test_X)  


    return (train_X,test_X)