# ECE1513 - Graduate Student Project
# Dylan Johnston - 1003852690
# Artificial Intelligence in Financial Markets
# April 10th, 2019

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Import Packages
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import keras
import fix_yahoo_finance as yf
from pandas.plotting import register_matplotlib_converters
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import mixture as mix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from random import shuffle
from pathlib import Path

register_matplotlib_converters()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Main Program Execution

def main():
    global df_raw, data_set, data_set_scaled, df_regime_partial, df_regime_scaled_partial, df_regime_max, df_regime_scaled_max, df_SVM_max, df_LSTM_max
    # Input desired security to analyze, and time frame over which the security should be analyzed
    security_ticker = "SPY"
    start_date = "1999-01-01"
    end_date = "2019-04-09"

    start = time.time()

    # Import financial data from Yahoo Finance
    df_raw = dataImport(security_ticker)

    # Create new shifted OHLC columns, append data set with various financial metrics, crop data set to desired time frame
    data_set, data_set_scaled = processData(df_raw, start_date, end_date)

    # Define Hyperparameters of Models
    globalVariableDefine()

    # Perform unsupervised GMM trend identification
    df_regime_partial, df_regime_scaled_partial, df_regime_max, df_regime_scaled_max = regimeIdentify()

    # Implement SVM classifier Model
    # df_SVM_partial = SVMpred(df_regime_partial, df_regime_scaled_partial)
    df_SVM_max = SVMpred(df_regime_max, df_regime_scaled_max)
    # df_SVM_no_regime = SVMpred(data_set, data_set_scaled)

    # Implement LSTM predictive Model
    # df_LSTM_partial = LSTMpred(df_regime_partial, df_regime_scaled_partial)
    df_LSTM_max = LSTMpred(df_regime_max, df_regime_scaled_max)
    # df_LSTM_no_regime = LSTMpred(data_set, data_set_scaled)

    end = time.time()
    total_time = end - start

def globalVariableDefine():
    # Hyperparameters of optimization
    global hp_epochs, hp_batch_size, hp_dropout, hp_predict_split, hp_look_back, hp_predict_period, hp_GMM_components, hp_GMM_split
    hp_epochs = 500
    hp_batch_size = 400
    hp_dropout = 0.2
    hp_GMM_split = 0.7
    hp_predict_split = 0.8
    hp_look_back = 50
    hp_predict_period = 10
    hp_GMM_components = 4

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Data Import and Processing

def dataImport(security_ticker):
    df = pd.DataFrame()
    # Where the csv file will be saved
    df_directory = "/home/thurteen/Desktop/Graduate School/Semester 4/ECE1513/Graduate Project/Datasets/"
    file_name = security_ticker + ".csv"
    path = df_directory + file_name
    # data_file = Path(path)

    # Retrieve data from Yahoo Finance
    df = yf.download(security_ticker)
    # Not using Adjusted Closes
    df = df.drop(columns = ['Adj Close'])
    # Save the data as a csv
    df.to_csv(index = False, path_or_buf = path)
    return df

def processData(data, start_date, end_date):
    df = data.copy()
    # Offset OHLC data 1 day forward, so that each row has
    # the Close of the current day and the OHLC of the previous day
    df['Open_Shifted'] = df['Open'].shift(1)
    df['High_Shifted'] = df['High'].shift(1)
    df['Low_Shifted'] = df['Low'].shift(1)
    df['Close_Shifted'] = df['Close'].shift(1)
    df['Volume_Shifted'] = df['Volume'].shift(1)

    # Append data set with various financial trading indicators
    df = augmentData(df)

    df['Signal'] = 0
    df.loc[df['Daily_Return'] > 0, 'Signal'] = 1
    df.loc[df['Daily_Return'] < 0, 'Signal'] = -1

    df.index = df['Date']
    df = df.drop(['Date','Open','High','Low','Close','Volume'], axis = 1)

    # Drop rows that have NaN or NaT in them.
    # These rows are at the top of the data set since some financial metrics
    # require a minimum number of periods before they can be calculated
    df = df.dropna()

    # Crop data set to desired start and end date for training and testing
    df = df[start_date:end_date]

    df_scaled = df.copy()

    mms = MinMaxScaler(feature_range = (0,1))
    ss = StandardScaler()

    # MinMaxColumns = pd.Index(['Open_Shifted', 'High_Shifted', 'Low_Shifted', 'Close_Shifted', 'Volume_Shifted'], dtype = 'object')
    MinMaxColumns = pd.Index(['Open_Shifted', 'High_Shifted', 'Low_Shifted', 'Close_Shifted', 'Volume_Shifted','2_Week_RSI','Month_RSI'], dtype = 'object')

    StanScalColumns = pd.Index(['Week_Momentum', 'MACD', 'MACD_Signal', 'MACD_Crossover', 'Chaikin'], dtype = 'object')

    df_scaled[MinMaxColumns] = mms.fit_transform(df_scaled[MinMaxColumns])
    df_scaled[StanScalColumns] = ss.fit_transform(df_scaled[StanScalColumns])

    return df, df_scaled

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Augment data set with technical trading indicators

def augmentData(data):
    df = data.copy()
    df = returns(df)
    df = momentum(df)
    # df = bollingerBands(df)
    df = MACD(df)
    df = RSI(df)
    df = Chaikin(df)
    return df

def returns(data):
    # This will be the target column for the SVM and LSTM predictors
    # Uses log ratio of todays price over yesterdays, + => Long, - => Short
    Daily_Return = pd.Series(np.log(data['Close']/data['Close'].shift(1)), name = 'Daily_Return')
    # Weekly_Return = pd.Series(np.log(data['Close']/data['Close'].shift(5)), name = 'Weekly_Return')
    # Monthly_Return = pd.Series(np.log(data['Close']/data['Close'].shift(20)), name = 'Monthly_Return')

    data = data.join(Daily_Return)
    # data = data.join(Weekly_Return)
    # data = data.join(Monthly_Return)
    return data

def momentum(data):
    # Want previous day's momentum, so uses shifted close
    df = data
    day_momentum = pd.Series(df['Close_Shifted'].diff(1), name = 'Day_Momentum')
    week_momentum = pd.Series(df['Close_Shifted'].diff(5), name = 'Week_Momentum')
    month_momentum = pd.Series(df['Close_Shifted'].diff(20), name = 'Month_Momentum')

    # df = df.join(day_momentum)
    df = df.join(week_momentum)
    # df = df.join(month_momentum)
    return df

def bollingerBands(data):
    # Want previous day's Bollinger Bands, so uses shifted close
    mean_band = pd.Series(data['Close_Shifted'].rolling(20, min_periods = 20).mean())
    sd_band = pd.Series(data['Close_Shifted'].rolling(20, min_periods = 20).std())
    b1 = 4* sd_band / mean_band
    B1 = pd.Series(b1, name = 'BollingerBands_1')
    b2 = (data['Close_Shifted'] - mean_band + 2 * sd_band) / (4 * sd_band)
    B2 = pd.Series(b2, name = 'BollingerBands_2')

    data = data.join(B1)
    data = data.join(B2)
    return data

def MACD(data):
    # Want previous day's MACD, so uses shifted close
    EMA_12 = pd.Series(data['Close_Shifted'].ewm(span = 12, min_periods = 26).mean())
    EMA_26 = pd.Series(data['Close_Shifted'].ewm(span = 26, min_periods = 26).mean())
    MACD = pd.Series(EMA_12 - EMA_26, name = 'MACD')
    MACDsign = pd.Series(MACD.ewm(span = 9, min_periods = 9).mean(), name = 'MACD_Signal')
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACD_Crossover')

    data = data.join(MACD)
    data = data.join(MACDsign)
    data = data.join(MACDdiff)
    return data

def RSI(data):
    # Want previous day's RSI, so uses shifted close
    data = data.reset_index()
    i = 0
    Up_I = [0]
    Down_I = [0]
    while i+1 <= data.index[-1]:
        UpMove = data.loc[i + 1, 'High_Shifted'] - data.loc[i, 'High_Shifted']
        DownMove = data.loc[i, 'Low_Shifted'] - data.loc[i + 1, 'Low_Shifted']
        if UpMove > DownMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        Up_I.append(UpD)
        if DownMove > UpMove and DownMove > 0:
            DoD = DownMove
        else:
            DoD = 0
        Down_I.append(DoD)
        i = i + 1

    Up_I = pd.Series(Up_I)
    Down_I = pd.Series(Down_I)

    PosDI_short = pd.Series(Up_I.ewm(span = 10, min_periods = 10).mean())
    NegDI_short = pd.Series(Down_I.ewm(span = 10, min_periods = 10).mean())
    RSI_short = pd.Series(PosDI_short / (PosDI_short + NegDI_short), name = '2_Week_RSI')

    PosDI_long = pd.Series(Up_I.ewm(span = 20, min_periods = 20).mean())
    NegDI_long = pd.Series(Down_I.ewm(span = 20, min_periods = 20).mean())
    RSI_long = pd.Series(PosDI_long / (PosDI_long + NegDI_long), name = 'Month_RSI')

    data = data.join(RSI_short)
    data = data.join(RSI_long)
    return data

def Chaikin(data):
    # Want previous day's Chaikin Oscillator Value, so uses shifted close
    ad = (2 * data['Close_Shifted'] - data['High_Shifted'] - data['Low_Shifted']) / (data['High_Shifted'] - data['Low_Shifted']) * data['Volume_Shifted']
    Chaikin = pd.Series(ad.ewm(span = 3, min_periods = 3).mean() - ad.ewm(span = 10, min_periods = 10).mean(), name = 'Chaikin')
    data = data.join(Chaikin)
    return data

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Machine Learning Algorithms

def regimeIdentify():
    # Using the GMM for predictions
    df = data_set_scaled.copy()
    df = df.drop(['Signal'], axis = 1)
    split = int(hp_GMM_split*len(df))

    # Here we try and predict the regime of the test set samples, and the predicted samples are used for security price prediction going forward, resulting in a much reduced data set
    regimes_partial = mix.GaussianMixture(n_components = hp_GMM_components, covariance_type = "spherical", n_init = 100, random_state = 42)
    regimes_partial.fit(np.reshape(df[:split],(-1,df.shape[1])))
    regime_partial = regimes_partial.predict(np.reshape(df[split:],(-1,df.shape[1])))

    regime_df_partial = pd.DataFrame(regime_partial, columns = ['Regime'], index = df[split:].index)

    df_regime_partial = data_set.copy()
    df_regime_scaled_partial = data_set_scaled.copy()

    df_regime_partial = df_regime_partial.iloc[split:,:]
    df_regime_scaled_partial = df_regime_scaled_partial.iloc[split:,:]

    df_regime_partial = df_regime_partial.join(regime_df_partial, how = 'inner')
    df_regime_scaled_partial = df_regime_scaled_partial.join(regime_df_partial, how = 'inner')

    cum_return_market = pd.DataFrame(df_regime_partial.Daily_Return.cumsum())
    df_regime_partial['cumulative_market_return'] = cum_return_market


    # Using the GMM to label regimes for whole dataset, and using whole data set for security price prediction training, resulting in a much larger data set.
    regimes_max = mix.GaussianMixture(n_components = hp_GMM_components, covariance_type = "spherical", n_init = 100, random_state = 42)
    regimes_max.fit(np.reshape(df,(-1,df.shape[1])))
    regime_max = regimes_max.predict(np.reshape(df,(-1,df.shape[1])))

    regime_df_max = pd.DataFrame(regime_max, columns = ['Regime'], index = df.index)

    df_regime_max = data_set.copy()
    df_regime_scaled_max = data_set_scaled.copy()

    df_regime_max = df_regime_max.join(regime_df_max, how = 'inner')
    df_regime_scaled_max = df_regime_scaled_max.join(regime_df_max, how = 'inner')

    # Plotting to see what it looks like
    regime_type = range(hp_GMM_components)

    plt.figure(1)
    plt.clf()
    plt.scatter(df_regime_partial.index, df_regime_partial['cumulative_market_return'], c = df_regime_partial['Regime'], s = 5)
    plt.title('Cumulative Return of SPY with Trends', fontsize = 32)
    plt.xlabel('Date', fontsize = 30)
    plt.ylabel('Cumulative Return', fontsize = 30)
    plt.grid(which = 'both', axis = 'both')
    # plt.legend(ncol=1, loc='upper left', fontsize = 16)
    plt.show()
    df_regime_partial = df_regime_partial.drop(['cumulative_market_return'], axis = 1)
    for i in regime_type:
        print('Predicted Regime = ', i+1)
        print('Mean for Predicted Regime ', i+1, ' = ', regimes_partial.means_[i][0])
        print('Covariance for Predicted Regime ', i+1, ' = ', regimes_partial.covariances_[i])
        print()

        print('Evaluated Regime = ', i+1)
        print('Mean for Evaluated Regime ', i+1, ' = ', regimes_max.means_[i][0])
        print('Covariance for Evaluated Regime ', i+1, ' = ', regimes_max.covariances_[i])
        print()

    return df_regime_partial, df_regime_scaled_partial, df_regime_max, df_regime_scaled_max

def SVMpred(data, data_scaled):
    global metric_SVM_Accuracy
    df = data.copy()
    df_scaled = data_scaled.copy()

    split = int(hp_predict_split*len(data_scaled))
    x = df_scaled.drop(['Signal','Daily_Return'], axis =1)
    y = df_scaled['Signal']

    SVM_Classifier = SVC(class_weight = None, gamma = 'auto', tol = 0.00001)

    SVM_Classifier.fit(x[:split], y[:split])

    prediction_data = len(x) - split

    df['Prediction_Signal'] = 0
    df.iloc[-prediction_data:, df.columns.get_loc('Prediction_Signal')] = SVM_Classifier.predict(x[split:])
    df['Strategy_Return'] = df['Prediction_Signal']*df['Daily_Return']

    df['Cumulative_Strategy_Return'] = 0
    df['Cumulative_Market_Return'] = 0

    df.iloc[-prediction_data:, df.columns.get_loc('Cumulative_Strategy_Return')] = np.nancumsum(df['Strategy_Return'][-prediction_data:])
    df.iloc[-prediction_data:, df.columns.get_loc('Cumulative_Market_Return')] = np.nancumsum(df['Daily_Return'][-prediction_data:])

    metric_SVM_Accuracy = accuracy_score(df['Prediction_Signal'], df['Signal'])

    plt.figure(2)
    plt.clf()
    plt.plot(df['Cumulative_Strategy_Return'][-prediction_data:], color = 'g', label = 'Strategy Returns')
    plt.plot(df['Cumulative_Market_Return'][-prediction_data:], color = 'r', label = 'Market Returns')
    plt.legend(loc = 'best')
    plt.title('SVM Model Return vs Market Return', fontsize = 32)
    plt.xlabel('Date', fontsize = 30)
    plt.ylabel('Cumulative Return', fontsize = 30)
    plt.grid(which = 'both', axis = 'both')
    plt.show

    return df

def LSTMpred(data, data_scaled):
    global metric_test_mse, metric_train_mse
    x_train, y_train, x_test, y_test, final = processLSTM(data, data_scaled)
    model = buildModel(x_train)

    history = model.fit(x_train, y_train, epochs = hp_epochs, validation_split = 0.1, batch_size = hp_batch_size, shuffle = True)

    metric_test_mse=model.evaluate(x_test,y_test,verbose=0)
    metric_train_mse=model.evaluate(x_train,y_train,verbose=0)

    final = getScore(x_test, y_test, model, final, history)
    # return data_LSTM_scaled

def buildModel(data):
    # Initialising the RNN
    model = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 100, return_sequences = True, input_shape = (data.shape[1], data.shape[2])))
    model.add(Dropout(hp_dropout))

    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 100, return_sequences = False))
    model.add(Dropout(hp_dropout))

    # Adding the output layer
    model.add(Dense(units = 1))

    # Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

    return model

def processLSTM(data, data_scaled):
    df = data.copy()
    df_scaled = data_scaled.copy()

    length = len(df)-1
    split = int(hp_predict_split * len(df))

    final = pd.DataFrame(df['Signal'])
    final['Daily_Return'] = df['Daily_Return']

    df_scaled = df_scaled.drop(['Daily_Return','Signal'], axis = 1)

    numpy_signal = np.array(df['Signal']).astype(float)
    numpy_scaled = np.array(df_scaled)

    xdata = []
    ydata = []

    for i in range(hp_look_back, length):
        xdata.append(numpy_scaled[i - hp_look_back:i,:])
        ydata.append(numpy_signal[i])

    xdata, ydata = np.array(xdata), np.array(ydata)

    final = final.iloc[split+hp_look_back:-1,:]

    x_train = xdata[:split]
    y_train = ydata[:split]

    x_test = xdata[split:]
    y_test = ydata[split:]

    return x_train, y_train, x_test, y_test, final

def getScore(x_test, y_test, model, final, history):
    global metric_LSTM_accuracy
    final['Predictions'] = model.predict(x_test)
    final['Buy_Order'] = 0
    final.loc[final['Predictions'] > 0, 'Buy_Order'] = 1
    final.loc[final['Predictions'] < 0, 'Buy_Order'] = -1

    metric_LSTM_accuracy = accuracy_score(final['Signal'],final['Buy_Order'])

    final['Strategy_Return'] = final['Buy_Order']*final['Daily_Return']

    final['Cumulative_Strategy_Return'] = 0
    final['Cumulative_Market_Return'] = 0

    final.iloc[:, final.columns.get_loc('Cumulative_Strategy_Return')] = np.nancumsum(final['Strategy_Return'][:])
    final.iloc[:, final.columns.get_loc('Cumulative_Market_Return')] = np.nancumsum(final['Daily_Return'][:])

    plt.figure(3)
    plt.clf()
    plt.plot(final['Cumulative_Strategy_Return'][:], color = 'g', label = 'Strategy Returns')
    plt.plot(final['Cumulative_Market_Return'][:], color = 'r', label = 'Market Returns')
    plt.legend(loc = 'best')
    plt.title('LSTM Predicted Return vs Market Return', fontsize = 32)
    plt.xlabel('Date', fontsize = 30)
    plt.ylabel('Cumulative Return', fontsize = 30)
    plt.grid(which = 'both', axis = 'both')
    plt.show

    plt.figure(4)
    plt.clf()
    plt.plot(history.history['acc'], color = 'g', label = 'Training Accuracy')
    plt.plot(history.history['val_acc'], color = 'r', label = 'Validation Accuracy')
    plt.legend(loc = 'best')
    plt.title('LSTM Accuracy vs Epoch', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.ylabel('Accuracy', fontsize = 30)
    plt.grid(which = 'both', axis = 'both')
    plt.show

    plt.figure(5)
    plt.clf()
    plt.plot(history.history['loss'], color = 'g', label = 'Training Loss')
    plt.plot(history.history['val_loss'], color = 'r', label = 'Validation Loss')
    plt.legend(loc = 'best')
    plt.title('LSTM Loss vs Epoch', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.ylabel('Loss', fontsize = 30)
    plt.grid(which = 'both', axis = 'both')
    plt.show

    return final

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

main()
