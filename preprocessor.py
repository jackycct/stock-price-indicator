import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from loader import StockPriceLoader
from relative_strength_index import RSI

class Preprocessor():
    def convert_dataset(self, data, step_size=1):
        dataX, dataY = [], []
        for i in range(len(data) - step_size - 1):
            a = data[i:(i + step_size), :]
            dataX.append(a)
            dataY.append(data[i + step_size, -1:])
        return np.array(dataX), np.array(dataY)

    def split_data(self, dataset, percentage=0.67):
        train_size = int(len(dataset) * percentage)
        return dataset[0:train_size], dataset[train_size:len(dataset)]

    def normalize_data_with_minmaxscaler(scaler, data):
        results = {}
        for symbol in data:
            results[symbol] = pd.DataFrame(scaler.fit_transform(data[symbol]))
        return results

    def prepare_multistock_data_with_rsi(self, no_of_record=1000, look_back=15):
        print "Preparing data with look back {}".format(look_back)
        symbols = ['jpm']

        loader = StockPriceLoader()

        historical_data = loader.load_single_stock('jpm')

        # Calculate RSI for the target stock
        rsi_calculator = RSI()
        rsi = rsi_calculator.calculate(historical_data['Close'], 14)
        rsi = rsi.dropna()
        data = historical_data

        # Join as a new column
        data['rsi'] = rsi
        data = historical_data[15:]

        gs = loader.load_single_stock('gs')
        gs.columns = gs.columns.map(lambda x: 'gs_' + str(x))

        bac = loader.load_single_stock('bac')
        bac.columns = bac.columns.map(lambda x: 'bac_' + str(x))

        c = loader.load_single_stock('c')
        c.columns = c.columns.map(lambda x: 'c_' + str(x))

        data = data.join(gs)
        data = data.join(bac)
        data = data.join(c)
        data = data.dropna()
        # print data.head()

        if no_of_record != -1:
            data = data[:no_of_record]

        delta = (data - data.shift(1)) / data.shift(1)
        delta = delta.dropna()

        # move Close to last column
        close = delta['Close']
        del delta['Close']
        delta = delta.join(close)

        print "No of rows loaded {}".format(len(delta))
        # print delta.head()

        # normalize the dataset
        # print delta[:3]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        dataset = scaler.fit_transform(delta)
        no_of_features = dataset.shape[1]
        # print dataset[:3]
        # split into train and test sets
        preprocessor = Preprocessor()
        train, test = preprocessor.split_data(dataset)
        # print train[:3]
        # convert to dataset matrix
        trainX, trainY = preprocessor.convert_dataset(train, look_back)
        testX, testY = preprocessor.convert_dataset(test, look_back)
        # print trainY[:3]
        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], no_of_features))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], no_of_features))

        return trainX, trainY, testX, testY, data, delta