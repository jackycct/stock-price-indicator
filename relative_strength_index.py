import pandas as pd

class RSI():
    def calculate(self, series, period):
        delta = series.diff().dropna()
        delta = delta[1:]

        # Make the positive gains (up) and negative gains (down) Series
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        # Calculate the SMA
        roll_up = pd.rolling_mean(up, period)
        roll_down = pd.rolling_mean(down.abs(), period)

        # Calculate the RSI based on SMA
        RS = roll_up / roll_down
        RSI = 100.0 - (100.0 / (1.0 + RS))

        return RSI