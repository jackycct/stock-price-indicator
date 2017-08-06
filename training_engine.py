import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

class TrainingEngine():
    def create_model(self, trainX, trainY, epochs=100, features=1, look_back=30, firstLevel=4, secondLevel=1, \
                     loss='mse', optimizer='adam'):
        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(firstLevel, input_shape=(look_back, features), return_sequences=True))
        model.add(LSTM(secondLevel, input_shape=(look_back, features), return_sequences=False))

        model.add(Dense(16, activation="relu", kernel_initializer="uniform"))
        model.add(Dense(1, activation="linear", kernel_initializer="uniform"))

        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        model.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)

        return model

    def predict(self, model, data, delta, trainX, testX, trainY, testY, look_back=15):
        # make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        # normalize the dataset
        delta_close = delta['Close']
        delta_close = np.reshape(delta_close, (delta_close.shape[0], 1))
        delta_close.astype('float32')
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit_transform(delta_close)

        # invert predictions
        inverseTrainPredict = scaler.inverse_transform(trainPredict)
        inverseTrainY = scaler.inverse_transform(trainY)
        inverseTestPredict = scaler.inverse_transform(testPredict)
        inverseTestY = scaler.inverse_transform(testY)

        # Skip the look_back elements
        base = data.ix[:, 'Close']

        originalTrainY = np.empty_like(inverseTrainY)
        originalTrainPredict = np.empty_like(inverseTrainPredict)

        for x in range(0, len(inverseTrainY)):
            originalTrainY[x] = base[x + look_back] * (1 + inverseTrainY[x])
            originalTrainPredict[x] = base[x + look_back] * (1 + inverseTrainPredict[x])

        originalTestY = np.empty_like(inverseTestY)
        originalTestPredict = np.empty_like(inverseTestPredict)

        for x in range(0, len(inverseTestY)):
            originalTestY[x] = base[x + len(inverseTrainY) + look_back * 2] * (1 + inverseTestY[x])
            originalTestPredict[x] = base[x + len(inverseTrainY) + look_back * 2] * (1 + inverseTestPredict[x])

        return base[1:].values, originalTrainPredict[:, 0], originalTrainY[:, 0], originalTestPredict[:, 0], originalTestY[:, 0]
