import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

class Reporting():
    def calculate_rmse(self, trainPredict, inverse_trainY, testPredict, inverse_testY):
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(inverse_trainY, trainPredict))
        testScore = math.sqrt(mean_squared_error(inverse_testY, testPredict))

        return trainScore, testScore

    def print_results(self, data, trainPredict, trainY, testPredict, testY, look_back, start=0, end=-1):
        if (end == -1):
            end = len(data)

        # calculate root mean squared error
        trainScore, testScore = self.calculate_rmse(trainPredict, trainY, testPredict, testY)
        print('Train Score: %.2f RMSE' % (trainScore))
        print('Test Score: %.2f RMSE' % (testScore))

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(data)
        trainPredictPlot[:] = np.nan
        trainPredictPlot[look_back:len(trainPredict) + look_back] = trainPredict

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(data)
        testPredictPlot[:] = np.nan
        testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(data) - 1] = testPredict

        # plot baseline and predictions
        plt.figure(figsize=(20, 10))
        plt.plot(data[start:end])
        plt.plot(trainPredictPlot[start:end])
        plt.plot(testPredictPlot[start:end])
        plt.show()