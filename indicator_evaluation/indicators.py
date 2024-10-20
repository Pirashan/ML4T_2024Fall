"""
Indicators:
- Bollinger Bands
- RSI
- Stochastic Indicator
- MACD
- CCIw
"""

from util import get_data, plot_data
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

def author():
    return 'pravikumaran3'

def studygroup():
    return 'pravikumaran3'

def bollinger_bands(prices, symbol, window=20, num_std_dev=2):
    prices = prices[symbol]

    # Calculate the rolling mean and rolling standard deviation
    rm = prices.rolling(window=window).mean()
    rstd = prices.rolling(window=window).std()

    # Calculate the upper and lower Bollinger Bands
    upper_band = rm + (rstd * num_std_dev)
    lower_band = rm - (rstd * num_std_dev)
    bb_value = (prices - rm)/(2*rstd)
    bb_percent = ((prices - lower_band) / (upper_band - lower_band)) * 100

    # Plotting the Bollinger Bands
    fig = plt.figure(figsize=(14, 7))
    plt.plot(rm, label='SMA', color='blue')
    plt.plot(prices, label='Prices', color='orange')
    plt.plot(upper_band, label='Upper Bollinger Band', color='red')
    plt.plot(lower_band, label='Lower Bollinger Band', color='green')
    # plt.plot(bb_value, label='Bollinger Band Value', color='gray')
    plt.fill_between(prices.index, lower_band, upper_band, color='lightgrey', alpha=0.2)
    plt.title(f'{symbol} Bollinger Bands', fontsize=20)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Price', fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig("Bollinger_Bands.png")

    # Plotting the BB%
    fig = plt.figure(figsize=(14, 7))
    plt.plot(bb_percent, label='Bollinger Band %', color='blue')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.axhline(y=100, color='black', linestyle='--')
    plt.title('Bollinger Band %', fontsize=20)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('BB%', fontsize=16)
    plt.ylim(-30, 130)
    plt.legend()
    plt.grid()
    # plt.text(0.5, 0.5, "pravikumaran3@gatech.edu", fontsize=30, rotation=50, color='gray',
    #          alpha=0.5, ha='center', va='center', transform=fig.transFigure)
    plt.savefig("BB%.png")

    return bb_percent

def rsi (prices, symbol, lookback = 14):
    data = prices.copy()
    # price changes
    data['Price_Change'] = data[symbol].diff()
    # daily gains/losses separately (abs val)
    data['Gain'] = np.where(data['Price_Change'] > 0, data['Price_Change'], 0)
    data['Loss'] = np.where(data['Price_Change'] < 0, -data['Price_Change'], 0)
    # average gain/losses for lookback period
    data['AG'] = data['Gain'].rolling(window=lookback, min_periods=lookback).mean()
    data['AL'] = data['Loss'].rolling(window=lookback, min_periods=lookback).mean()
    # Relative Strength and Relative Strength Index
    data['RS'] = data['AG'] / data['AL']
    data['RSI'] = 100 - (100 / (1 + data['RS']))

    # Plotting the RSI
    fig = plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['RSI'], label=f'{symbol} RSI', color='blue')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title(f'{symbol} RSI (Lookback = 14)', fontsize=20)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('RSI', fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig("RSI.png")

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

    # Plot Stochastic
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['%K'], label=f'{symbol} %K (14)', color='blue')
    plt.plot(data.index, data['%D'], label=f'{symbol} %D (3)', color='red')
    # Highlight crossovers
    for i in range(1, len(crossover_signal)):
        if crossover_signal[i] == 1 and crossover_signal[i - 1] != 1:
            plt.plot(data.index[i], data['%K'][i], 'g^', markersize=10)  # Buy signal
        elif crossover_signal[i] == -1 and crossover_signal[i - 1] != -1:
            plt.plot(data.index[i], data['%K'][i], 'rv', markersize=10)  # Sell signal
    plt.axhline(80, color='black', linestyle='--', label='Overbought (80)')
    plt.axhline(20, color='black', linestyle='--', label='Oversold (20)')
    plt.title(f'{symbol} Stochastic Oscillator', fontsize=20)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Stochastic Value', fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig("Stochastic.png")

    return crossover_signal

def macd(prices, symbol, short_period = 12, long_period= 26, signal_period = 9):
    data = prices.copy()
    #short-term EMA (12-day)
    short_term_ema = data[symbol].ewm(span=short_period, min_periods=1).mean()

    #long-term EMA (26-day)
    long_term_ema = data[symbol].ewm(span=long_period, min_periods=1).mean()

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

    # Plot MACD
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, macd_results['MACD Line'], label=f'{symbol} MACD (12, 26)', color='blue', linewidth=1.5)
    plt.plot(data.index, macd_results['Signal Line'], label=f'{symbol} Signal (9)', color='red', linestyle='--', linewidth=1.5)
    positive_hist = macd_results['Histogram'] >= 0
    negative_hist = macd_results['Histogram'] < 0
    plt.bar(data.index[positive_hist], macd_results['Histogram'][positive_hist], color='green', alpha=0.6)
    plt.bar(data.index[negative_hist], macd_results['Histogram'][negative_hist], color='red', alpha=0.6)
    plt.title(f'{symbol} MACD', fontsize=20)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('MACD Value', fontsize=16)
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.grid()
    plt.savefig("MACD.png")

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

    # Plotting the CCI
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, cci_df['CCI'], label=f'{symbol} CCI (20)', color='blue')
    plt.axhline(100, color='red', linestyle='--', label='Overbought (+100)')
    plt.axhline(-100, color='green', linestyle='--', label='Oversold (-100)')
    plt.title(f'{symbol} CCI (Lookback = 20)', fontsize=20)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('CCI Value', fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig("CCI.png")

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

    # RSI
    data = rsi(prices, symbol, lookback = 14)

    # Stochastic
    data = stochastic_indicator(prices, symbol, lookback_k=14, lookback_d=3)

    # MACD
    data = macd(prices, symbol, short_period = 12, long_period= 26, signal_period = 9)

    # CCI
    data = cci(prices, symbol, lookback=20, high_prices=high_prices, low_prices=low_prices, close_prices=close_prices)


if __name__ == "__main__":
    test_code()