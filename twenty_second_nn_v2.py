import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
import sqlite3
import numpy as np
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report


db_file = '/Users/Shayan1/Desktop/DataProject/twenty_second_order_book_data_v2.db'
conn = sqlite3.connect(db_file, timeout=1200)


def create_lstm_regression_dataset(local_df, time_steps):
    features = local_df.loc[:, local_df.columns != 'trade_signal'].to_numpy()
    target   = local_df[['trade_signal']].to_numpy()
    x = []
    y = []
    
    for i in range(time_steps-1, len(features)):
        temp = []
        for j in range(time_steps):
            temp.append(features[i-j])
        x.append(temp)
        y.append(target[i])
        
    return numpy.array(x), numpy.array(y)

def create_lstm_categorical_dataset(local_df, time_steps):
    features, target = create_dataset(local_df)
    x = []
    y = []
    
    for i in range(time_steps-1, len(features)):
        temp = []
        for j in range(time_steps):
            temp.append(features[i-j])
        #print('x', temp, 'y', target[i])
        
        x.append(temp)
        y.append(target[i])
        
    return numpy.array(x), numpy.array(y)

def create_dataset(local_df):
    features = local_df.loc[:, local_df.columns != 'trade_signal'].to_numpy()
    target = []
    for i in local_df.index:
        if (local_df.loc[i, 'trade_signal'] == -1):
            target.append([1.0, 0.0])
        if (local_df.loc[i, 'trade_signal'] ==  1):
            target.append([0.0, 1.0])
 
    target = numpy.array(target)
    #dataset = tf.data.Dataset.from_tensor_slices((features.values, target))
    return features, target


def lstm_regression_training(local_df, time_steps, symbol):
    # Transform data from dataframe into numpy array that can be fed into keras/tensor flow
    train_size = int(.8*len(local_df))
    test_size  = len(local_df) - train_size
    
    x_train, y_train = create_lstm_regression_dataset(local_df[:train_size], time_steps)
    x_test,  y_test  = create_lstm_regression_dataset(local_df[train_size:], time_steps)
   
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(20)
    test_dataset  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(20)
    
    print(x_train.shape)
    print(y_train.shape)
    print()

    # Build neural network with keras and tensorflow
    model = keras.models.Sequential()
    model.add(LSTM(100, return_sequences=(True), input_shape=((x_train.shape[1],x_train.shape[2]))))
    model.add(LSTM(10))
    #model.add(Flatten(input_shape=(x_train.shape[1],x_train.shape[2])))
    #model.add(keras.layers.Dense(200, activation="relu"))
    #model.add(keras.layers.Dense(100, activation="relu"))
    #model.add(keras.layers.Dense(50, activation="relu"))
    #model.add(keras.layers.Dense(10, activation="relu"))
    model.add(keras.layers.Dense(1, activation="linear"))
    # mean_absolute_error
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss = "mean_squared_error", 
                  optimizer = opt)
    
    filepath=symbol+"weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    # Fit the model
    history = model.fit(train_dataset,
                        validation_data=(test_dataset),
                        epochs=50,
                        shuffle=False, 
                        callbacks=callbacks_list, 
                        verbose=2)
    model.summary()
    
    model.load_weights(filepath)
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(symbol+' loss vs val_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    print()
    
    temp_list = []
    y_true = []
    y_predicted = []
    for i in range(int(test_size)-time_steps):
        y_pred = model.predict(x_test[i].reshape(1, x_train.shape[1],x_train.shape[2])) 
        
        print (y_pred[0][0], y_test[i][0], i)
        y_true.append(y_test[i][0])
        y_predicted.append(y_pred[0][0])
        
    plt.scatter(y_predicted,y_true)
    plt.title(symbol)
    plt.show()
    return model
    
    
def lstm_categorical_training(local_df, time_steps, symbol):
    # Transform data from dataframe into numpy array that can be fed into keras/tensor flow
    train_size = int(.8*len(local_df))
    test_size  = len(local_df) - train_size
    
    x_train, y_train = create_lstm_categorical_dataset(local_df[:train_size], time_steps)
    x_test,  y_test  = create_lstm_categorical_dataset(local_df[train_size:], time_steps)
    num_categories   = len(y_train[0])
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(25)
    test_dataset  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(25)
    
    print(x_train.shape)
    print(y_train.shape)
    print()
    
    # Build neural network with keras and tensorflow
    model = keras.models.Sequential()
    #model.add(LSTM(10, return_sequences=(True), input_shape=((x_train.shape[1],x_train.shape[2]))))
    #model.add(LSTM(5))
    model.add(Dropout(0.1))
    model.add(Flatten(input_shape=(x_train.shape[1],x_train.shape[2])))
    model.add(keras.layers.Dense(200, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(50, activation="relu"))
    model.add(keras.layers.Dense(10, activation="sigmoid"))
    model.add(keras.layers.Dense(num_categories, activation="sigmoid"))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss = "binary_crossentropy", 
                  optimizer = opt, 
                  metrics = ["accuracy"])
    
    filepath=symbol+"weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    # Fit the model
    history = model.fit(train_dataset,
                        validation_data=(test_dataset),
                        epochs=1000,
                        shuffle=False, 
                        callbacks=callbacks_list, 
                        verbose=2)
    model.summary()
    
    model.load_weights(filepath)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(symbol +' model accuracy')
    plt.xlabel('epoch')
    plt.legend(['train acc', 'val acc'], loc='upper left')
    plt.show()
    
    temp_list = []
    y_true = []
    y_predicted = []
    
    cumulative_return = 0
    cumulative_return_tc = 0
    
    cumulative_return_list = []
    cumulative_return_tc_list = []
    for i in range(int(test_size-time_steps)):
        y_pred = model.predict(x_test[i].reshape(1, x_train.shape[1],x_train.shape[2])) 
        
        if (y_pred[0][0]>y_pred[0][1]):
            y_predict = -1
        else:
            y_predict = 1
        
        if (y_test[i][0] == 1):
            y_actual = -1
        elif (y_test[i][1] == 1):
            y_actual = 1
            
        if (y_pred[0][0]>.9):
            position = -1
        elif (y_pred[0][1]>.9):
            position = 1
        else:
            position = 0
        
        actual_return = x_test[i+1][0][0]
        cumulative_return = cumulative_return+(position*(actual_return-1)*100)
        cumulative_return_list.append(cumulative_return)
        
        print ('{:>2} {:>2} {:>3} {:>5} {:>15} {:>5} {:>5}'.format(
                position, 
                y_actual, 
                y_predict, 
                round(actual_return, 3), 
                str(y_pred.round(2)), 
                str(y_test[i].round(3)),
                round(cumulative_return, 3), 
                i))
        
        y_true.append(y_actual)
        y_predicted.append(y_predict)
    
    #target_names=['Super Short', 'Short', 'No Trend', 'Long', 'Super Long']
    print(classification_report(y_true, y_predicted, target_names=['Short', 'Long',]))
    ##print(classification_report(y_true, y_strong, target_names=['Short', 'No Trend', 'Long']))
    plt.plot(cumulative_return_list)
    plt.title('Cumulative Return: '+symbol)
    plt.show()
    
    return model
    
# This function processes order book data as a percentage volume of total orders 
#   between specified depth levels away from the mark price
def data_preprocessing_from_dataframe1(df):
    df2 = pd.DataFrame()
    
    df2['price_last'] = df['price_last']
    # Scale desired features if necessary for Neural network from original dataframe
    scaler = MinMaxScaler()
    df2['previous_1period_return']   = np.log(df['price_last']/df['price_last'].shift(1))+1
    df2['bid_ask_spread']       = df['bid_ask_spread']
    df2['price_high_minus_low'] = scaler.fit_transform(df[['price_high_minus_low']])
    df2['bid_volume']           = scaler.fit_transform(df[['bid_volume']])
    df2['ask_volume']           = scaler.fit_transform(df[['ask_volume']])
    df2['bid_trades']           = scaler.fit_transform(df[['bid_trades']])
    df2['ask_trades']           = scaler.fit_transform(df[['ask_trades']])
    
    # Convert bids and asks in order book at specific depth level to percentages of total order book
    # All numbers must be between 0 and 1 so already scaled. 
    
    total_bids = 'total_bids_within_'
    total_asks = 'total_asks_within_'
    df2['bid%_between_0_100_bps'] = (df['total_bids_within_100_bps']/
                                    df['total_orders_order_book'])
    df2['ask%_between_0_100_bps'] = (df['total_asks_within_100_bps']/
                                    df['total_orders_order_book'])
    for i in range (2, 11):
        df2['bid%_between_'+str((i-1)*100)+'_'+str(i*100)+'_bps'] = (
            (df['total_bids_within_'+str(i*100)+'_bps']-
             df['total_bids_within_'+str((i-1)*100)+'_bps'])/
             df['total_orders_order_book'])
        df2['ask%_between_'+str((i-1)*100)+'_'+str(i*100)+'_bps'] = (
            (df['total_asks_within_'+str(i*100)+'_bps']-
             df['total_asks_within_'+str((i-1)*100)+'_bps'])/
             df['total_orders_order_book'])
        
    return df2

# This function processes order book data as a percentage volume of total orders 
#   at a specified depth level away from the mark price
def data_preprocessing_from_dataframe2(df):
    df2 = pd.DataFrame()
    
    # Scale desired features if necessary for Neural network from original dataframe
    scaler = MinMaxScaler()
    df2['previous_1period_return']   = np.log(df['price_last']/df['price_last'].shift(1))+1
    
    #df2['bid_ask_spread']       = df['bid_ask_spread']
    df2['price_high_minus_low'] = scaler.fit_transform(df[['price_high_minus_low']])
    df2['bid_volume']           = scaler.fit_transform(df[['bid_volume']])
    df2['ask_volume']           = scaler.fit_transform(df[['ask_volume']])
    #df2['bid_trades']           = scaler.fit_transform(df[['bid_trades']])
    #df2['ask_trades']           = scaler.fit_transform(df[['ask_trades']])
    
    # Convert bids and asks in order book at specific depth level to cumulative percentages of total order book
    # All numbers must be between 0 and 1 so already scaled. 
    
    for i in range (1, 11):
        df2['bid%_within_'+str(i*100)+'_bps'] = (
             df['total_bids_within_'+str(i*100)+'_bps']/
             df['total_orders_order_book'])
        df2['ask%_within_'+str(i*100)+'_bps'] = (
             df['total_asks_within_'+str(i*100)+'_bps']/
             df['total_orders_order_book'])
        
    return df2

# This function processes order book data as a normalized volume  
#   between specified depth level away from the mark price
def data_preprocessing_from_dataframe3(df):
    df2 = pd.DataFrame()
    
    # Scale desired features if necessary for Neural network from original dataframe
    scaler = MinMaxScaler()
    df2['previous_1period_return']   = np.log(df['price_last']/df['price_last'].shift(1))+1
    df2['bid_ask_spread']       = df['bid_ask_spread']
    df2['price_high_minus_low'] = scaler.fit_transform(df[['price_high_minus_low']])
    df2['bid_volume']           = scaler.fit_transform(df[['bid_volume']])
    df2['ask_volume']           = scaler.fit_transform(df[['ask_volume']])
    df2['bid_trades']           = scaler.fit_transform(df[['bid_trades']])
    df2['ask_trades']           = scaler.fit_transform(df[['ask_trades']])
    
    # Calculate number of order bewteen specific depth level
    total_bids = 'total_bids_within_'
    total_asks = 'total_asks_within_'
    df2['bid%_between_0_10_bps'] = (df['total_bids_within_10_bps']/
                                    df['total_orders_order_book'])
    df2['ask%_between_0_10_bps'] = (df['total_asks_within_10_bps']/
                                    df['total_orders_order_book'])
    for i in range (2, 101):
        df2['bid%_between_'+str((i-1)*10)+'_'+str(i*10)+'_bps'] = (
            df['total_bids_within_'+str(i*10)+'_bps']-
            df['total_bids_within_'+str((i-1)*10)+'_bps'])
        df2['ask%_between_'+str((i-1)*10)+'_'+str(i*10)+'_bps'] = (
            df['total_asks_within_'+str(i*10)+'_bps']-
            df['total_asks_within_'+str((i-1)*10)+'_bps'])
        
    # Normalize number of orders using MinMaxScaler
    df2['bid%_between_0_10_bps'] = scaler.fit_transform(df2[['bid%_between_0_10_bps']])
    df2['ask%_between_0_10_bps'] = scaler.fit_transform(df2[['ask%_between_0_10_bps']])
    
    for i in range (2, 101):
        df2['bid%_between_'+str((i-1)*10)+'_'+str(i*10)+'_bps'] = (
            scaler.fit_transform(df2[['bid%_between_'+str((i-1)*10)+'_'+str(i*10)+'_bps']]))
        df2['ask%_between_'+str((i-1)*10)+'_'+str(i*10)+'_bps'] = (
            scaler.fit_transform(df2[['ask%_between_'+str((i-1)*10)+'_'+str(i*10)+'_bps']]))
        
    return df2

def transform_data_to_longer_timeframe(df, num_periods):
    df2 = pd.DataFrame()
    df2['symbol'] = df['symbol']
    df2['datetime'] = df['datetime']
    
    df2['price_last'] = df['price_last']
    df2['low_price'] = df['low_price'].rolling(num_periods).min()
    df2['high_price'] = df['high_price'].rolling(num_periods).max()
    df2['price_high_minus_low']    = df['high_price'] - df['low_price']
    
    
    df2['bid_volume']   = df['bid_volume'].rolling(num_periods).sum()
    df2['ask_volume']   = df['ask_volume'].rolling(num_periods).sum()
    df2['total_volume'] = df['total_volume'].rolling(num_periods).sum() 
    
    df2['bid_dollar_volume'] = df['bid_dollar_volume'].rolling(num_periods).sum()
    df2['ask_dollar_volume'] = df['ask_dollar_volume'].rolling(num_periods).sum()
    df2['dollar_volume'] = df['dollar_volume'].rolling(num_periods).sum()
    
    df2['bid_trades'] = df['bid_trades'].rolling(num_periods).sum()
    df2['ask_trades'] = df['ask_trades'].rolling(num_periods).sum()
    df2['num_trades'] = df['num_trades'].rolling(num_periods).sum()
    
    df2['bid_ask_spread'] = df['bid_ask_spread'].rolling(num_periods).mean()
    
    total_bids = 'total_bids_within_'
    total_asks = 'total_asks_within_'
    for i in range (1, 101):
        df2[total_bids+str((i)*10)+'_bps'] = df[total_bids+str((i)*10)+'_bps'].rolling(num_periods).mean()
        df2[total_asks+str((i)*10)+'_bps'] = df[total_asks+str((i)*10)+'_bps'].rolling(num_periods).mean()
        #df2[total_bids+str((i)*10)+'_bps'] = df[total_bids+str((i)*10)+'_bps']
        #df2[total_asks+str((i)*10)+'_bps'] = df[total_asks+str((i)*10)+'_bps']
    
    
    df2['total_orders_order_book'] = df['total_orders_order_book'].rolling(num_periods).mean()
    
    df2 = df2.iloc[::num_periods, :]
    df2 = df2.dropna()
    return df2
        
symbol_list = ['AAVEUSDT',
               'AVAXUSDT',
               'BCHUSDT',
               'BNBUSDT',
               'BTCUSDT',
               'DOTUSDT',
               'EOSUSDT',
               'ETHUSDT',
               'FILUSDT',
               'KSMUSDT',
               'LINKUSDT',
               'LTCUSDT',
               'MKRUSDT',
               'SUSHIUSDT',
               'TRBUSDT',
               'UNIUSDT',
               'WAVESUSDT',
               'XMRUSDT',
               'ZECUSDT']  
   
symbol_list = ['LTCUSDT']

for symbol in symbol_list:    
    df = pd.read_sql_query('SELECT * FROM '+symbol, con = conn)
    df = df.dropna()
    
    # Extra columns which will be used to calculate features for neural network
    df['total_orders_order_book'] = (df['total_bids_within_1000_bps']+
                                     df['total_asks_within_1000_bps'])
    
    num_periods = 15
    
    df2 = transform_data_to_longer_timeframe(df, num_periods)
    
    df3 = data_preprocessing_from_dataframe1(df2)
    
    run_regression = False
    if (run_regression):
        #df3['trade_signal'] = df3['previous_1m_return']*100
        df3['trade_signal'] = df3['previous_1period_return'].shift(-1)
        #print(df3.groupby('trade_signal').count())
        df3 = df3.dropna()
        time_steps = 99
        model = lstm_categorical_training(df3, time_steps, symbol)
        #model = lstm_categorical_training(df3, time_steps)
        #model = neural_net_training(df3) 
    
        
    run_categorical = True
    if (run_categorical):
        # Calculate trade signal. This will be the desired output of neural network
        # Go long (1)  if there is a sufficient return for long  position (.001)
        # Go short(-1) if there is a sufficient return for short position (-.001)
    
        df3.loc[ (df3['previous_1period_return'].shift(-1) > 1), 'trade_signal'] =  1
        #df3.loc[((df2['forward_1period_return']<=  .0022) &
        #         (df2['forward_1period_return']>=  .0005)), 'trade_signal'] =  1
        #df3.loc[((df2['forward_1period_return']<=  .0005) &
        #         (df2['forward_1period_return']>= -.0005)), 'trade_signal'] =  0
        #df3.loc[((df2['forward_1period_return']<= -.0005) &
        #         (df2['forward_1period_return']>= -.0022)), 'trade_signal'] =  -1
        df3.loc[ (df3['previous_1period_return'].shift(-1) < 1), 'trade_signal'] = -1  
        
        print(df3.groupby('trade_signal').count())
        df3 = df3.dropna()
        time_steps = 1
        model = lstm_categorical_training(df3, time_steps, symbol)
        