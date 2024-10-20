import datetime as dt
import os
import numpy as np
import pandas as pd
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util import get_data, plot_data
# pd.set_option('display.max_rows', None)

def author():
    return 'pravikumaran3'

def studygroup():
    return 'pravikumaran3'

def testPolicy(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000):
    #get prices
    prices = get_data([symbol], pd.date_range(sd, ed))
    prices = prices.drop(columns="SPY")

    trades = pd.DataFrame(0, index = prices.index, columns = ['Price','Symbol', 'Order', 'Trades'])

    for symbol in [symbol]:
        symbol_prices = prices[symbol]

        for i in range(len(symbol_prices)):
            trades.loc[symbol_prices.index[i], 'Price'] = symbol_prices[i]
            trades.loc[symbol_prices.index[i], 'Symbol'] = symbol

            if i + 1 >= len(symbol_prices):
                continue

            # Buy or Sell 1000 on the first day
            if i == 0:
                if symbol_prices[i] < symbol_prices[i + 1]:  # BUY condition
                    trades.loc[symbol_prices.index[i], 'Order'] = 'BUY'
                    trades.loc[symbol_prices.index[i], 'Trades'] = 1000
                elif symbol_prices[i] > symbol_prices[i + 1]:  # SELL condition
                    trades.loc[symbol_prices.index[i], 'Order'] = 'SELL'
                    trades.loc[symbol_prices.index[i], 'Trades'] = -1000
                continue

            # For following days
            if symbol_prices[i] < symbol_prices[i + 1] and symbol_prices[i] <= symbol_prices[i - 1]:
                # Current day is BUY
                trades.loc[symbol_prices.index[i], 'Order'] = 'BUY'

                # If the previous day was also a BUY, it should now be a HOLD (0 shares)
                if trades.loc[symbol_prices.index[i - 1], 'Order'] == 'BUY':
                    trades.loc[symbol_prices.index[i], 'Order'] = 'HOLD'
                    trades.loc[symbol_prices.index[i], 'Trades'] = 0  # Hold
                else:
                    trades.loc[symbol_prices.index[i], 'Trades'] = 2000

            elif symbol_prices[i] > symbol_prices[i + 1] and symbol_prices[i] >= symbol_prices[i - 1]:
                # Current day is SELL
                trades.loc[symbol_prices.index[i], 'Order'] = 'SELL'

                # If the previous also a SELL, it's now a HOLD (0 shares)
                if trades.loc[symbol_prices.index[i - 1], 'Order'] == 'SELL':
                    trades.loc[symbol_prices.index[i], 'Order'] = 'HOLD'
                    trades.loc[symbol_prices.index[i], 'Trades'] = 0  # Hold
                else:
                    trades.loc[symbol_prices.index[i], 'Trades'] = -2000

    # Filter out trades
    trades = trades[trades['Trades'] != 0]
    trades = trades[['Trades']]

    return trades

def main():
    #call to testPolicy
    trades = testPolicy(symbol= "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    # print(trades)
    # print(trades.head())

if __name__ == "__main__":
    main()