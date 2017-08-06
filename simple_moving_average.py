class SMA():
    def get_rolling_mean(self, values, window):
        return values.rolling(window=window).mean()

    def get_sma(self, data, normalise_window):
        sma = {}
        for symbol in data:
            sma[symbol] = self.get_rolling_mean(data[symbol], normalise_window)
        return sma