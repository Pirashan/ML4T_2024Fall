import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import StrategyLearner as sl
import ManualStrategy
import marketsimcode as msc

np.random.seed(903948218)

def author():
    return 'pravikumaran3'

def stats(portvals):
    daily_returns = (portvals / portvals.shift(1)) - 1
    dr = daily_returns[1:]
    cr = (portvals[-1] / portvals[0]) - 1  # cumulative return
    mdr = dr.mean()  # mean of daily return
    sddr = dr.std()  # stdev of daily return
    return cr, mdr, sddr

def table():
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    impact = 0.005
    commission = 9.95

    # Manual Strategy Portfolio
    ms = ManualStrategy.ManualStrategy(symbol)
    df_trades, _, _ = ms.testPolicy(symbol, sd, ed, sv)
    df_trades['Date'] = pd.to_datetime(df_trades['Date'])
    df_trades.set_index('Date', inplace=True)

    ms_portvals = msc.compute_portvals(df_trades, sd, ed, sv, 9.95, 0.005)
    ms_portvals = ms_portvals / ms_portvals.iloc[0]
    #Statistics
    cr1, mdr1, sddr1 = stats(ms_portvals)

    # Benchmark Portolio
    bench_portvals = ms.benchmark(symbol, sd, ed, sv)
    bench_portvals = bench_portvals / bench_portvals.iloc[0]
    #Statistics
    cr2, mdr2, sddr2 = stats(bench_portvals)

    # Strategy Learner Portfolio
    learner = sl.StrategyLearner(verbose=False, impact=0.005, commission=commission)
    learner.add_evidence(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df = learner.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df = build_trades(trades_df, symbol)
    trades_df['Date'] = pd.to_datetime(trades_df['Date'])
    trades_df.set_index('Date', inplace=True)
    learner_portvals = msc.compute_portvals(trades_df, sd, ed, sv, 9.95, 0.005)
    learner_portvals = learner_portvals / learner_portvals.iloc[0]
    # Statistics
    cr3, mdr3, sddr3 = stats(learner_portvals)

    # print(f"Manual Cumulative Return: {cr1}")
    # print(f"Benchmark Cumulative Return: {cr2}")
    # print(f"Strategy Cumulative Return: {cr3}")
    # print(f"Manual Mean Daily Return: {mdr1}")
    # print(f"Benchmark Mean Daily Return: {mdr2}")
    # print(f"Strategy Mean Daily Return: {mdr3}")
    # print(f"Manual Std Dev of Daily Return: {sddr1}")
    # print(f"Benchmark Std Dev of Daily Return: {sddr2}")
    # print(f"Strategy Std Dev of Daily Return: {sddr3}")

    return cr1, cr2, cr3, mdr1, mdr2, mdr3, sddr1, sddr2, sddr3


def build_trades(prices, symbol):
    trades = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
    index = 0
    for i in range(0, prices.shape[0]):
        # Access the price in the 'JPM' column for each row
        price = prices.iloc[i][symbol]

        # Now compare the price value
        if price == 2000:
            trades.loc[index] = [prices.index[i].strftime('%Y-%m-%d'), symbol, 'BUY', 2000]
            index = index + 1
        elif price == 1000:
            trades.loc[index] = [prices.index[i].strftime('%Y-%m-%d'), symbol, 'BUY', 1000]
            index = index + 1
        elif price == -2000:
            trades.loc[index] = [prices.index[i].strftime('%Y-%m-%d'), symbol, 'SELL', 2000]
            index = index + 1
        elif price == -1000:
            trades.loc[index] = [prices.index[i].strftime('%Y-%m-%d'), symbol, 'SELL', 1000]
            index = index + 1

    return trades

def plots():
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    impact = 0.005
    commission = 9.95

    ms = ManualStrategy.ManualStrategy(symbol)
    df_trades,_,_ = ms.testPolicy(symbol, sd, ed, sv)
    df_trades['Date'] = pd.to_datetime(df_trades['Date'])
    df_trades.set_index('Date', inplace=True)
    # print(df_trades)
    ms_portvals = msc.compute_portvals(df_trades, sd, ed, sv, 9.95, impact)
    ms_portvals = ms_portvals / ms_portvals.iloc[0]

    bench_portvals = ms.benchmark(symbol, sd, ed, sv)
    bench_portvals = bench_portvals / bench_portvals.iloc[0]

    learner = sl.StrategyLearner(verbose=False, impact=0.005, commission=commission)
    learner.add_evidence(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df = learner.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df = build_trades(trades_df, symbol)
    trades_df['Date'] = pd.to_datetime(trades_df['Date'])
    trades_df.set_index('Date', inplace=True)

    learner_portvals = msc.compute_portvals(trades_df, sd, ed, sv, 9.95, 0.005)
    learner_portvals = learner_portvals / learner_portvals.iloc[0]

    plt.figure(figsize=(18, 9))
    plt.plot(ms_portvals, 'b', label='Manual Strategy')
    plt.plot(bench_portvals, 'g', label='Benchmark')
    plt.plot(learner_portvals, 'r', label='Strategy Learner')
    plt.xticks(rotation=13, fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=20)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Normalized Price', fontsize=20)
    plt.title('Normalized Value of Manual vs Benchmark vs Strategy Learner In-Sample', fontsize=26)
    plt.savefig('Experiment1_In_Sample.png')
    plt.close()

    # Out Of Sample Plot
    os_sd = dt.datetime(2010, 1, 1)
    os_ed = dt.datetime(2011, 12, 31)

    ms = ManualStrategy.ManualStrategy(symbol)
    df_trades, _, _ = ms.testPolicy(symbol, os_sd, os_ed, sv)
    df_trades['Date'] = pd.to_datetime(df_trades['Date'])
    df_trades.set_index('Date', inplace=True)

    ms_portvals = msc.compute_portvals(df_trades, os_sd, os_ed, sv, 9.95, 0.005)
    ms_portvals = ms_portvals / ms_portvals.iloc[0]

    bench_portvals = ms.benchmark(symbol, os_sd, os_ed, sv)
    bench_portvals = bench_portvals / bench_portvals.iloc[0]

    learner = sl.StrategyLearner(verbose=False, impact=0.005, commission = commission)
    learner.add_evidence(symbol="JPM", sd=os_sd, ed=os_ed, sv=sv)
    trades_df = learner.testPolicy(symbol="JPM", sd=os_sd, ed=os_ed, sv=sv)
    trades_df = build_trades(trades_df, symbol)
    trades_df['Date'] = pd.to_datetime(trades_df['Date'])
    trades_df.set_index('Date', inplace=True)

    learner_portvals = msc.compute_portvals(trades_df, os_sd, os_ed, sv, 9.95, 0.005)
    learner_portvals = learner_portvals / learner_portvals.iloc[0]

    plt.figure(figsize=(18, 9))
    plt.plot(ms_portvals, 'b', label='Manual Strategy')
    plt.plot(bench_portvals, 'g', label='Benchmark')
    plt.plot(learner_portvals, 'r', label='Strategy Learner')
    plt.xticks(rotation=13, fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=20)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Normalized Price', fontsize=20)
    plt.title('Normalized Value of Manual vs Benchmark vs Strategy Learner Out-of-Sample', fontsize=26)
    plt.savefig('Experiment1_Out_of_Sample.png')
    plt.close()

if __name__ == "__main__":
    plots()