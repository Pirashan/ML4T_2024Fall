import datetime as dt
import random
import pandas as pd
import util
import numpy as np
import matplotlib.pyplot as plt

import indicators
import marketsimcode as msc

np.random.seed(903948218)
random.seed(903948218)

class ManualStrategy(object):
    # constructor
    def __init__(self, symbol, verbose=False, impact=0.005, commission=9.95):
        """
        Constructor method
        """
        self.symbol = symbol
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def author(self):
        return 'pravikumaran3'

    def testPolicy(self, symbol, sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        dates = pd.date_range(sd, ed)
        extended_dates = pd.date_range(pd.to_datetime(sd) - pd.Timedelta(days=50), ed)

        extended_prices = util.get_data([symbol], extended_dates) #automatically adds SPY
        extended_prices = extended_prices[symbol] #drop SPY
        prices_all = util.get_data([symbol], dates)  # automatically adds SPY
        prices = prices_all[symbol]  # only portfolio symbols, drop SPY

        # Get all indicators, moving_window = x days
        bbp_data = indicators.bollinger_bands(extended_prices, symbol, window=16)
        rsi_data = indicators.rsi(extended_prices, symbol, lookback=18)
        macd_data = indicators.macd(extended_prices, symbol, short_period = 12, long_period= 26, signal_period = 9)
        # Restrict indicators to the original date range
        bbp_data = bbp_data.loc[sd:ed]
        rsi_data = rsi_data.loc[sd:ed]
        macd_data = macd_data.loc[sd:ed]
        # pd.set_option('display.max_rows', None)  # Display all rows
        # print("BBP Data:\n", bbp_data)
        # print("RSI Data:\n", rsi_data)
        # print("MACD Data:\n", macd_data)

        trades = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
        index = 0
        buydate = []
        selldate = []
        position = 0  # position 1 = have shares, position 0 = no share, position -1 = shorted

        for i in range(prices.shape[0] - 1):
            if position == 0:
                if (
                        bbp_data.iloc[i] < 30
                        and rsi_data.iloc[i] < 30
                        or macd_data.iloc[i] < -0.8
                ):
                    trades.loc[index] = [prices.index[i + 1].strftime('%Y-%m-%d'), symbol, 'BUY', 1000]
                    position = 1
                    index += 1
                    buydate.append(prices.index[i + 1].date())
                elif (
                        bbp_data.iloc[i] > 70
                        and rsi_data.iloc[i] > 70
                        or macd_data.iloc[i] > 0.7
                ):
                    trades.loc[index] = [prices.index[i + 1].strftime('%Y-%m-%d'), symbol, 'SELL', 1000]
                    position = -1
                    index += 1
                    selldate.append(prices.index[i + 1].date())
            elif position == -1:
                if (
                        bbp_data.iloc[i] < 20
                        and rsi_data.iloc[i] < 20
                        or macd_data.iloc[i] < -0.5
                ):
                    trades.loc[index] = [prices.index[i + 1].strftime('%Y-%m-%d'), symbol, 'BUY', 2000]
                    position = 1
                    index += 1
                    buydate.append(prices.index[i + 1].date())
                elif (
                        bbp_data.iloc[i] < 25
                        and rsi_data.iloc[i] < 20
                        or macd_data.iloc[i] < -0.65
                ):
                    trades.loc[index] = [prices.index[i + 1].strftime('%Y-%m-%d'), symbol, 'BUY', 1000]
                    position = 0
                    index += 1
            elif position == 1:
                if (
                        bbp_data.iloc[i] > 90
                        and rsi_data.iloc[i] > 70
                        or macd_data.iloc[i] > 0.8
                ):
                    trades.loc[index] = [prices.index[i + 1].strftime('%Y-%m-%d'), symbol, 'SELL', 2000]
                    position = -1
                    index += 1
                    selldate.append(prices.index[i + 1].date())
                elif (
                        bbp_data.iloc[i] > 70
                        or rsi_data.iloc[i] > 60
                        or macd_data.iloc[i] > 0.6
                ):
                    trades.loc[index] = [prices.index[i + 1].strftime('%Y-%m-%d'), symbol, 'SELL', 1000]
                    position = 0
                    index += 1

        if position == 1:
            trades.loc[index] = [prices.index[-1].strftime('%Y-%m-%d'), symbol, 'SELL', 1000]
        if position == -1:
            trades.loc[index] = [prices.index[-1].strftime('%Y-%m-%d'), symbol, 'BUY', 1000]

        return trades, buydate, selldate

    def benchmark(self, symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        # Starting with $100,000 cash, invest in 1000 shares of JPM and then hold that position.
        df_benchmark,_,_ = self.testPolicy(symbol, sd, ed, sv)
        df_benchmark['Date'] = pd.to_datetime(df_benchmark['Date'])  # Ensure the column is in datetime format
        df_benchmark.set_index('Date', inplace=True)  # Set the 'Date' column as the index
        df_benchmark.iloc[0, 1] = 'BUY'
        for i in range(1, df_benchmark.shape[0] - 1):
            df_benchmark.iloc[i, 1] = np.NaN
        df_benchmark.dropna(inplace=True)
        bench_portvals = msc.compute_portvals(df_benchmark, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), 100000, 9.95, 0.005)
        return bench_portvals

    def stats(self, portvals):
        daily_returns = (portvals / portvals.shift(1)) - 1
        dr = daily_returns[1:]
        cr = (portvals[-1] / portvals[0]) - 1  # cumulative return
        mdr = dr.mean()  # mean of daily return
        sddr = dr.std()  # stdev of daily return
        return cr, mdr, sddr

    def table(self):
        symbol = 'JPM'
        start_val = 100000

        # In Sample Period
        df_trades1,_,_ = self.testPolicy(symbol, dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), start_val)
        df_trades1['Date'] = pd.to_datetime(df_trades1['Date'])
        df_trades1.set_index('Date', inplace=True)
        portvals1 = msc.compute_portvals(df_trades1, dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), start_val, 9.95, 0.005)
        portvals1 = portvals1 / portvals1.iloc[0]
        cr1, mdr1, sddr1 = self.stats(portvals1)

        bench_portvals1 = self.benchmark(symbol, dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), start_val)
        bench_portvals1 = bench_portvals1 / bench_portvals1.iloc[0]
        cr2, mdr2, sddr2 = self.stats(bench_portvals1)

        # Out of Sample Period
        df_trades2,_,_ = self.testPolicy(symbol, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), start_val)
        df_trades2, _, _ = self.testPolicy(symbol, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), start_val)
        df_trades2['Date'] = pd.to_datetime(df_trades2['Date'])
        df_trades2.set_index('Date', inplace=True)
        portvals2 = msc.compute_portvals(df_trades2, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), start_val, 9.95, 0.005)
        portvals2 = portvals2 / portvals2.iloc[0]
        cr3, mdr3, sddr3 = self.stats(portvals2)

        bench_portvals2 = self.benchmark(symbol, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), start_val)
        bench_portvals2 = bench_portvals2 / bench_portvals2.iloc[0]
        cr4, mdr4, sddr4 = self.stats(bench_portvals2)
        return cr1, cr2, cr3, cr4, mdr1, mdr2, mdr3, mdr4, sddr1, sddr2, sddr3, sddr4

    # Compare the performance of Manual Strategy versus the benchmark for in-sample and out-of-sample
    def plot_in_sample(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                       sv=100000):
        # Normalize the benchmark portfolio
        benchmark_vals = self.benchmark(symbol, sd, ed, sv)
        benchmark_vals = benchmark_vals / benchmark_vals.iloc[0]

        # Fetch trades and compute portfolio values
        trades_data, entry_points, exit_points = self.testPolicy(symbol, sd, ed, sv)
        trades_data['Date'] = pd.to_datetime(trades_data['Date'])
        trades_data.set_index('Date', inplace=True)
        strategy_vals = msc.compute_portvals(trades_data, sd, ed, sv, 9.95, 0.005)
        strategy_vals = strategy_vals / strategy_vals.iloc[0]

        # Plotting
        fig, ax = plt.subplots(figsize=(18, 9))
        plt.plot(strategy_vals, 'r', label="Manual Strategy")
        plt.plot(benchmark_vals, 'purple', label="Benchmark")
        plt.title("Manual Strategy vs. Benchmark: In-Sample", fontsize=26)
        plt.xlabel("Date", fontsize=20)
        plt.xticks(rotation=13, fontsize=18)
        plt.ylabel("Normalized Portfolio Value", fontsize=20)
        plt.yticks(fontsize=18)

        # Mark entry and exit points
        for date in entry_points:
            plt.axvline(date, color='b', linestyle='--', label="LONG")
        for date in exit_points:
            plt.axvline(date, color='k', linestyle='--', label="SHORT")

        # Remove duplicate labels in the legend
        handles, labels = ax.get_legend_handles_labels()
        unique_handles, unique_labels = [], []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_handles.append(handle)
                unique_labels.append(label)
        plt.legend(unique_handles, unique_labels, fontsize=18)
        plt.savefig('Manual_In_Sample.png')
        plt.close()

    def plot_out_of_sample(self, symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31),
                           sv=100000):
        # Normalize the benchmark portfolio
        benchmark_vals = self.benchmark(symbol, sd, ed, sv)
        benchmark_vals = benchmark_vals / benchmark_vals.iloc[0]

        # Fetch trades and compute portfolio values
        trades_data, entry_points, exit_points = self.testPolicy(symbol, sd, ed, sv)
        trades_data['Date'] = pd.to_datetime(trades_data['Date'])
        trades_data.set_index('Date', inplace=True)
        strategy_vals = msc.compute_portvals(trades_data, sd, ed, sv, 9.95, 0.005)
        strategy_vals = strategy_vals / strategy_vals.iloc[0]

        # Plotting
        fig, ax = plt.subplots(figsize=(18, 9))
        plt.plot(strategy_vals, 'r', label="Manual Strategy")
        plt.plot(benchmark_vals, 'purple', label="Benchmark")
        plt.title("Manual Strategy vs. Benchmark: Out-Of-Sample", fontsize=26)
        plt.xlabel("Date", fontsize=20)
        plt.xticks(rotation=13, fontsize=18)
        plt.ylabel("Normalized Portfolio Value", fontsize=20)
        plt.yticks(fontsize=18)

        # Mark entry and exit points
        for date in entry_points:
            plt.axvline(date, color='b', linestyle='--', label="LONG")
        for date in exit_points:
            plt.axvline(date, color='k', linestyle='--', label="SHORT")

        # Remove duplicate labels in the legend
        handles, labels = ax.get_legend_handles_labels()
        unique_handles, unique_labels = [], []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_handles.append(handle)
                unique_labels.append(label)
        plt.legend(unique_handles, unique_labels, fontsize=18)
        plt.savefig('Manual_Out_Of_Sample.png')
        plt.close()

if __name__ == "__main__":
    # Create an instance of the ManualStrategy class
    manual_strategy = ManualStrategy(symbol="JPM", verbose=True)

    # print in-sample performance
    manual_strategy.plot_in_sample()

    # print out-of-sample performance
    manual_strategy.plot_out_of_sample()

    # print performance stats
    cr1, cr2, cr3, cr4, mdr1, mdr2, mdr3, mdr4, sddr1, sddr2, sddr3, sddr4 = manual_strategy.table()
    print(f"In-Sample Cumulative Return: {cr1}, Benchmark Cumulative Return: {cr2}")
    print(f"Out-of-Sample Cumulative Return: {cr3}, Benchmark Cumulative Return: {cr4}")
    print(f"In-Sample Mean Daily Return: {mdr1}, Benchmark Mean Daily Return: {mdr2}")
    print(f"Out-of-Sample Mean Daily Return: {mdr3}, Benchmark Mean Daily Return: {mdr4}")
    print(f"In-Sample Std Dev of Daily Return: {sddr1}, Benchmark Std Dev of Daily Return: {sddr2}")
    print(f"Out-of-Sample Std Dev of Daily Return: {sddr3}, Benchmark Std Dev of Daily Return: {sddr4}")
