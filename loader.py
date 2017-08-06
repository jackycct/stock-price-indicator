import pandas as pd

class StockPriceLoader():
    def load_single_stock(self, symbol):
        history = pd.read_csv("data/daily/table_{}.csv".format(symbol), delimiter=',',
                              header=None,
                              index_col='Date', parse_dates=['Date'],
                              names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
                              usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        return history

    def load_multiple_stock(self, symbols):
        historical_price = {}

        for s in symbols:
            historical_price[s] = self.load_single_stock(s)

        return historical_price
