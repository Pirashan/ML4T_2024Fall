"""
Indicators:
- Bollinger Bands
- RSI
- Stochastic Indicator
- MACD
- CCIw
"""

from util import get_data
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

def author():
    return 'pravikumaran3'

def studygroup():
    return 'pravikumaran3'

def bollinger_bands(prices, symbol, window=20, num_std_dev=2):
    # Calculate the rolling mean and rolling standard deviation
    rm = prices.rolling(window=window).mean()
    rstd = prices.rolling(window=window).std()
    # Calculate the upper and lower Bollinger Bands
    upper_band = rm + (rstd * num_std_dev)
    lower_band = rm - (rstd * num_std_dev)
    bb_value = (prices - rm)/(2*rstd)
    bb_percent = ((prices - lower_band) / (upper_band - lower_band)) * 100

    return bb_percent

def rsi (prices, symbol, lookback = 14):
    data = prices.copy()
    if isinstance(data, pd.Series):  # Check if it's a Series (single column)
        data = data.to_frame()
    # price changes
    data['Price_Change'] = data.diff()
    # daily gains/losses separately (abs val)
    data['Gain'] = np.where(data['Price_Change'] > 0, data['Price_Change'], 0)
    data['Loss'] = np.where(data['Price_Change'] < 0, -data['Price_Change'], 0)
    # average gain/losses for lookback period
    data['AG'] = data['Gain'].rolling(window=lookback, min_periods=lookback).mean()
    data['AL'] = data['Loss'].rolling(window=lookback, min_periods=lookback).mean()
    # Relative Strength and Relative Strength Index
    data['RS'] = data['AG'] / data['AL']
    data['RSI'] = 100 - (100 / (1 + data['RS']))

    return data['RSI']

def stochastic_indicator(prices, symbol, lookback_k=14, lookback_d=3):
    data = prices.copy()
    #Calculating K
    data['Low'] = data[symbol].rolling(window=lookback_k).min()
    data['High'] = data[symbol].rolling(window=lookback_k).max()
    data['%K'] = 100 * (data[symbol] - data['Low']) / (data['High'] - data['Low'])

    #Calculating D
    data['%D'] = data['%K'].rolling(window=lookback_d).mean()

    #Crossover Calculation
    crossover_signal = np.where(data['%K'] > data['%D'], 1, 0)  # 1 if %K is above %D, 0 if not
    crossover_signal = np.where(data['%K'] < data['%D'], -1, crossover_signal)  # -1 if %K is below %D
    crossover_signal = pd.Series(crossover_signal, index=data.index).fillna(0)  # Fill NaN values with 0

    return crossover_signal

def macd(prices, symbol, short_period = 12, long_period= 26, signal_period = 9):
    data = prices.copy()
    if isinstance(data, pd.Series):  # Check if it's a Series (single column)
        data = data.to_frame()
    #short-term EMA (12-day)
    short_term_ema = data.ewm(span=short_period, min_periods=1).mean()

    #long-term EMA (26-day)
    long_term_ema = data.ewm(span=long_period, min_periods=1).mean()

    #MACD line
    macd_line = short_term_ema - long_term_ema

    #Signal line
    signal_line = macd_line.ewm(span=signal_period, min_periods=1).mean()

    #MACD histogram
    macd_histogram = macd_line - signal_line

    #results
    macd_results = pd.DataFrame(index=data.index)
    macd_results['MACD Line'] = macd_line
    macd_results['Signal Line'] = signal_line
    macd_results['Histogram'] = macd_histogram
    macd_results = macd_results['Histogram']
    return macd_results

def cci(prices, symbol, lookback=20, high_prices=None, low_prices=None, close_prices=None):
    data = prices.copy()
    #adjust high and low prices
    adjusted_close = prices[symbol]
    unadjusted_close = close_prices[symbol]

    # Calculate the adjustment ratio
    adjustment_ratio = adjusted_close / unadjusted_close

    # Adjust high and low prices
    adjusted_high = high_prices[symbol] * adjustment_ratio
    adjusted_low = low_prices[symbol] * adjustment_ratio

    # Calculate the typical price
    tp = (high_prices[symbol] + low_prices[symbol] + prices[symbol]) / 3

    # rolling mean of TP
    tp_mean = tp.rolling(window=lookback).mean()

    # mean deviation
    tp_mean_dev = tp.rolling(window=lookback).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)

    # CCI calculation
    cci = (tp - tp_mean) / (0.015 * tp_mean_dev)

    # Results
    cci_df = pd.DataFrame(index=prices.index)
    cci_df['CCI'] = cci

    return cci_df

def test_code():
    # Fetching data for JPM
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    symbol = 'JPM'

    dates = pd.date_range(start_date, end_date)
    prices = get_data([symbol], dates)
    high_prices = get_data([symbol], dates, colname="High")
    low_prices = get_data([symbol], dates, colname="Low")
    close_prices = get_data([symbol], dates, colname="Close")
    prices = prices.drop(columns="SPY")

    # Bollinger Bands
    bb_percent = bollinger_bands(prices, symbol)
    print(bb_percent)

    # RSI
    data = rsi(prices, symbol, lookback = 14)
    print(data)

    # Stochastic
    data = stochastic_indicator(prices, symbol, lookback_k=14, lookback_d=3)
    print(data)
    # MACD
    data = macd(prices, symbol, short_period = 12, long_period= 26, signal_period = 9)
    print(data)
    # CCI
    data = cci(prices, symbol, lookback=20, high_prices=high_prices, low_prices=low_prices, close_prices=close_prices)
    print(data)

if __name__ == "__main__":
    test_code()