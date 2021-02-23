import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
import sqlite3
import seaborn as sns
import numpy as np
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error


db_file = '/Users/Shayan1/Desktop/DataProject/minute_data_copy.db'
conn = sqlite3.connect(db_file, timeout=1200)


def create_dataset(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps-1):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps-1])
    return np.array(Xs), np.array(ys)

def lstm_training(dataframe):
    print (len(dataframe))
    dataframe = dataframe.iloc[1:]
    print (len(dataframe))
    dataframe = dataframe[['minute_return', 
                           'volume', 
                           'number_of_trades',
                           'bid_ask_ratio_20_bps',
                           'bid_ask_ratio_50_bps',
                           'bid_ask_ratio_100_bps',
                           'bid_ask_ratio_200_bps',
                           'numeric_indicator']]
    
    train_size = int(len(dataframe) * 0.6)
    test_size = len(dataframe) - train_size
    train, test = dataframe.iloc[0:train_size], dataframe.iloc[train_size:len(dataframe)]
    
    f_columns = ['minute_return', 
                 'volume', 
                 'number_of_trades',
                 'bid_ask_ratio_20_bps',
                 'bid_ask_ratio_50_bps',
                 'bid_ask_ratio_100_bps',
                 'bid_ask_ratio_200_bps']

    f_transformer = RobustScaler()
    f_transformer = f_transformer.fit(train[f_columns].to_numpy())
    train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
    test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
    
    #print (train)
    time.sleep(2.0)
    #indicator_transformer = RobustScaler()    
    #indicator_transformer = indicator_transformer.fit(train[['numeric_indicator']])    
    #train['numeric_indicator'] = indicator_transformer.transform(train[['numeric_indicator']])    
    #test['numeric_indicator']  = indicator_transformer.transform(test[['numeric_indicator']])
    
    train_numeric_indicator = train['numeric_indicator']
    test_numeric_indicator  = test['numeric_indicator']
    
    print(train)
    
    train = train[f_columns]
    test  = test[f_columns]
    
    time_steps = 60
    # reshape to [samples, time_steps, n_features]    
    X_train, y_train = create_dataset(train, train_numeric_indicator, time_steps)
    X_test, y_test =   create_dataset(test, test_numeric_indicator , time_steps)    
    
    
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    X_test  = np.asarray(X_test).astype(np.float32)
    y_test  = np.asarray(y_test).astype(np.float32)
    
    #print (X_train)
    #print (y_train)
    
    print (X_train.shape)
    print (y_train.shape)
    
    model = keras.Sequential()
    model.(LSTM(units=60,
                   input_shape=(X_train.shape[1], 
                                X_train.shape[2])))
    model.add(keras.layers.Dropout(rate=0.0001))
    model.add(keras.layers.Dense(units=1, activation='linear'))
    optimizer = tf.keras.optimizers.Adam(0.0001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=10,
        validation_split=0.2,
        shuffle=False)
    
    print()
    print("Evaluate on test data")
    results = model.evaluate(X_train, y_train)
    #results = model.evaluate(X_test, y_test)
    print("test acc:", results)
    
    #test model
    
    print("prediction vs truth:")
    
    y_hat = model.predict(X_train)
    y_train2 = []
    for i in y_train:
        temp = []
        temp.append(i)
        y_train2.append(temp)
    for i in range(0,20):
        print (str(y_hat[i]) + " " + str(y_train2[i])) 
    plt.scatter(y_train2, y_hat)
    
    
    y_hat_test = model.predict(X_test)
    y_test2 = []
    for i in y_test:
        temp = []
        temp.append(i)
        y_test2.append(temp)
    plt.scatter(y_test2, y_hat_test)
    
    # print ("Correlation: " + str(np.corrcoef(y_train2, y_hat)))
    print(model.summary())
    print()
    time.sleep(60.0)
    
    
symbol_list = ['AAVEUSDT', 
               'BCHUSDT',
               'BNBUSDT',
               'DOTUSDT',
               'ETHUSDT',
               'FILUSDT',
               'LINKUSDT',
               'LTCUSDT']
for symbol in symbol_list:    
    df = pd.read_sql_query('SELECT * FROM '+symbol, con = conn)

    df['bid_ask_ratio_20_bps'] = df['bid_ask_ratio_20_bps'].clip(upper=2)
    df['bid_ask_ratio_50_bps'] = df['bid_ask_ratio_50_bps'].clip(upper=2)
    df['bid_ask_ratio_100_bps'] = df['bid_ask_ratio_100_bps'].clip(upper=2)
    df['bid_ask_ratio_200_bps'] = df['bid_ask_ratio_200_bps'].clip(upper=2)
    
    df['minute_return']     = np.log((df['close_price']/df['close_price'].shift(1)))
    df['minute_volatility'] =  df['minute_return'].rolling(30).std()
    
    df['1_stddev_move'] = df['minute_volatility']*math.sqrt(10)*1
    df['2_stddev_move'] = df['minute_volatility']*math.sqrt(10)*2
    df['3_stddev_move'] = df['minute_volatility']*math.sqrt(10)*3
    
    df['forward_return_0_to_1_mins']  = df['minute_return'].shift(-1)
    df['forward_return_1_to_2_mins']  = df['minute_return'].shift(-2)
    df['forward_return_2_to_3_mins']  = df['minute_return'].shift(-3)
    df['forward_return_3_to_4_mins']  = df['minute_return'].shift(-4)
    df['forward_return_4_to_5_mins']  = df['minute_return'].shift(-5)
    df['forward_return_5_to_6_mins']  = df['minute_return'].shift(-6)
    df['forward_return_6_to_7_mins']  = df['minute_return'].shift(-7)
    df['forward_return_7_to_8_mins']  = df['minute_return'].shift(-8)
    df['forward_return_8_to_9_mins']  = df['minute_return'].shift(-9)
    df['forward_return_9_to_10_mins'] = df['minute_return'].shift(-10)
    
    #Number of up movements.
    df['number_of_up_movements']   = 0
    df['number_of_down_movements'] = 0
    
    df.loc[df['forward_return_0_to_1_mins']> 0, 'number_of_up_movements'] = df['number_of_up_movements']+1
    df.loc[df['forward_return_1_to_2_mins']> 0, 'number_of_up_movements'] = df['number_of_up_movements']+1
    df.loc[df['forward_return_2_to_3_mins']> 0, 'number_of_up_movements'] = df['number_of_up_movements']+1
    df.loc[df['forward_return_3_to_4_mins']> 0, 'number_of_up_movements'] = df['number_of_up_movements']+1
    df.loc[df['forward_return_4_to_5_mins']> 0, 'number_of_up_movements'] = df['number_of_up_movements']+1
    df.loc[df['forward_return_5_to_6_mins']> 0, 'number_of_up_movements'] = df['number_of_up_movements']+1
    df.loc[df['forward_return_6_to_7_mins']> 0, 'number_of_up_movements'] = df['number_of_up_movements']+1
    df.loc[df['forward_return_7_to_8_mins']> 0, 'number_of_up_movements'] = df['number_of_up_movements']+1
    df.loc[df['forward_return_8_to_9_mins']> 0, 'number_of_up_movements'] = df['number_of_up_movements']+1
    df.loc[df['forward_return_9_to_10_mins']>0, 'number_of_up_movements'] = df['number_of_up_movements']+1
    
    df.loc[df['forward_return_0_to_1_mins']< 0, 'number_of_down_movements'] = df['number_of_down_movements']+1
    df.loc[df['forward_return_1_to_2_mins']< 0, 'number_of_down_movements'] = df['number_of_down_movements']+1
    df.loc[df['forward_return_2_to_3_mins']< 0, 'number_of_down_movements'] = df['number_of_down_movements']+1
    df.loc[df['forward_return_3_to_4_mins']< 0, 'number_of_down_movements'] = df['number_of_down_movements']+1
    df.loc[df['forward_return_4_to_5_mins']< 0, 'number_of_down_movements'] = df['number_of_down_movements']+1
    df.loc[df['forward_return_5_to_6_mins']< 0, 'number_of_down_movements'] = df['number_of_down_movements']+1
    df.loc[df['forward_return_6_to_7_mins']< 0, 'number_of_down_movements'] = df['number_of_down_movements']+1
    df.loc[df['forward_return_7_to_8_mins']< 0, 'number_of_down_movements'] = df['number_of_down_movements']+1
    df.loc[df['forward_return_8_to_9_mins']< 0, 'number_of_down_movements'] = df['number_of_down_movements']+1
    df.loc[df['forward_return_9_to_10_mins']<0, 'number_of_down_movements'] = df['number_of_down_movements']+1
    
    df['forward_return_0_to_10_mins'] = (df['forward_return_0_to_1_mins'] +
                                         df['forward_return_1_to_2_mins'] +
                                         df['forward_return_2_to_3_mins'] +
                                         df['forward_return_3_to_4_mins'] +
                                         df['forward_return_4_to_5_mins'] +
                                         df['forward_return_5_to_6_mins'] +
                                         df['forward_return_6_to_7_mins'] +
                                         df['forward_return_7_to_8_mins'] +
                                         df['forward_return_8_to_9_mins'] +
                                         df['forward_return_9_to_10_mins'])
    
    
    df['average_bid_ask_ratio'] = (df['bid_ask_ratio_20_bps'] +
                                   df['bid_ask_ratio_50_bps'] +
                                   df['bid_ask_ratio_100_bps'] +
                                   df['bid_ask_ratio_200_bps'])/4 
   
    #df['indicator'] = 'No Trade' 
    #df.loc[(((df['forward_return_0_to_10_mins'] >  df['1_stddev_move']) & (df['number_of_up_movements']>=2))), 'indicator'] = 'LONG'
    #df.loc[(((df['forward_return_0_to_10_mins'] < -df['1_stddev_move']) & (df['number_of_down_movements']>=2))), 'indicator'] = 'SHORT'
    #df.loc[(df['minute_return'] > 0), 'indicator'] = 'LONG'
    #df.loc[(df['minute_return'] < 0), 'indicator'] = 'SHORT'
    
    
    df['numeric_indicator'] = df['forward_return_0_to_10_mins']
    
    #df.loc[(df['indicator']=='LONG'), 'numeric_indicator'] = 1.0
    #df.loc[(df['indicator']=='No Trade'), 'numeric_indicator'] = 0.0
    #df.loc[(df['indicator']=='SHORT'), 'numeric_indicator'] = -1.0
    
    df2 = df[['symbol',
              'close_time', 
              'minute_return', 
              'volume', 
              'number_of_trades',
              'bid_ask_ratio_20_bps',
              'bid_ask_ratio_50_bps',
              'bid_ask_ratio_100_bps',
              'bid_ask_ratio_200_bps',
              'forward_return_0_to_10_mins',
              'numeric_indicator']]
    lstm_training(dataframe = df2)



'''
date = df['close_time']
close_price = df['close_price']
df['bid_ask_ratio_20_bps'] = df['bid_ask_ratio_20_bps'].clip(upper=4)
df['bid_ask_ratio_50_bps'] = df['bid_ask_ratio_50_bps'].clip(upper=4)
df['bid_ask_ratio_100_bps'] = df['bid_ask_ratio_100_bps'].clip(upper=4)
df['bid_ask_ratio_200_bps'] = df['bid_ask_ratio_200_bps'].clip(upper=4)

mean_of_order_book = df.loc[: , "bid_ask_ratio_20_bps":"bid_ask_ratio_200_bps"].mean(axis=1)

rolling_mean_order_book = mean_of_order_book.rolling(10).mean()
rolling_mean_close_price = close_price.rolling(10).mean()

# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(date, rolling_mean_close_price, color="red")
# set x-axis label
ax.set_xlabel("time",fontsize=14)
# set y-axis label
ax.set_ylabel("Price",color="red",fontsize=14)

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(date, rolling_mean_order_book,color="blue")
ax2.set_ylabel("Bid/Ask Ratio",color="blue",fontsize=14)
plt.show()
'''
