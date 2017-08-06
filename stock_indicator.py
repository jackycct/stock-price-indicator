from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error

class StockIndicator():
    def stock_history_reader(self, symbol):
        history = pd.read_csv("data/daily/table_{}.csv".format(symbol), delimiter=',',
                              header=None,
                              index_col='Date', parse_dates=['Date'],
                              names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
                              usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        return history

    def load_data(self, symbols):
        historical_price = {}

        for s in symbols:
            historical_price[s] = self.stock_history_reader(s)

        return historical_price

    def get_rolling_mean(self, values, window):
        return values.rolling(window=window).mean()

    def get_sma(self, data, normalise_window):
        sma = {}
        for symbol in data:
            sma[symbol] = self.get_rolling_mean(data[symbol], normalise_window)
        return sma

    def normalize_data_with_minmaxscaler(self, scaler, data):
        results = {}
        for symbol in data:
            results[symbol] = pd.DataFrame(scaler.fit_transform(data[symbol]))
        return results

    def normalize_data_with_sma(self, data, mean):
        results = {}
        for symbol in data:
            results[symbol] = data[symbol][1:] / mean[symbol][1:] - 1
            results[symbol] = results[symbol].dropna(axis='index')
        return results

    def dataset_generate(self, data, step_size=1):
        dataX, dataY = [], []
        for i in range(len(data) - step_size - 1):
            a = data[i:(i + step_size)]
            dataX.append(a)
            dataY.append(data[i + step_size])
        return np.array(dataX), np.array(dataY)

    def split_data(self, dataset):
        train_size = int(len(dataset) * 0.75)
        return dataset[0:train_size], dataset[train_size:len(dataset)]

    def create_model(self, trainX, trainY, look_back):
        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, loodk_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)

        return model

    def __init__(self):
        symbols = ['jpm', 'gs']
        normalise_window = 30
        step_size = 1
        # fix random seed for reproducibility
        np.random.seed(7)

        historical_data = self.load_data(symbols)
        sma = self.get_sma(historical_data, normalise_window)
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        #normalized = self.normalize_data_with_minmaxscaler(historical_data)

        normalized = self.normalize_data_with_sma(historical_data, sma)

        print historical_data['jpm'].ix[29:34]
        print normalized['jpm'].head()
        test = (normalized['jpm'] + 1) * sma['jpm'].dropna(axis='index')
        print test.head()

        #close_price = self.normalized['jpm'].ix[0:200, 'Close']
        close_price = normalized['jpm']['Close']
        train, test = self.split_data(close_price)
        trainX, trainY = self.dataset_generate(train, step_size)
        testX, testY = self.dataset_generate(test, step_size)

        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        model = self.create_model(trainX, trainY, step_size)

        testPredict = model.predict(testX)
        print "Test Prediction"
        print testPredict
        print "SMA"
        print sma['jpm']['Close'].dropna(axis='index')
        testPredict += 1
        testPredict = np.multiply(testPredict, sma['jpm']['Close'].dropna(axis='index'))
        print testPredict


def main():
    indicator = StockIndicator()
    #print indicator.symbols
    #print indicator.historical_data['jpm'].head()
    indicator.historical_data['jpm']['Close'].plot()
    plt.show()
    #'.date_range('2008-01-01', '2008-12-31').head()

if __name__ == "__main__":
    main()

