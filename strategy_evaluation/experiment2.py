import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import StrategyLearner as sl
import marketsimcode as msc

def author():
    return 'pravikumaran3'

def stats(portvals):
    daily_returns = (portvals / portvals.shift(1)) - 1
    dr = daily_returns[1:]
    cr = (portvals[-1] / portvals[0]) - 1  # cumulative return
    mdr = dr.mean()  # mean of daily return
    sddr = dr.std()  # stdev of daily return
    return cr, mdr, sddr

def build_trades(prices, symbol):
    trades = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
    index = 0
    for i in range(0, prices.shape[0]):
        price = prices.iloc[i][symbol]

        # compare the price value
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

def table():
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    commission = 0.00

    # impact = 0.00
    learner = sl.StrategyLearner(verbose=False, impact=0)
    learner.add_evidence(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df = learner.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    learner_portvals = msc.compute_portvals(trades_df, sd, ed, sv, 0, 0)
    learner_portvals = learner_portvals / learner_portvals.iloc[0]
    cr1, mdr1, sddr1 = stats(learner_portvals)
    # impact = 0.005
    learner0 = sl.StrategyLearner(verbose=False, impact=0)
    learner0.add_evidence(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df0 = learner0.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    learner_portvals0 = msc.compute_portvals(trades_df0, sd, ed, sv, 0, 0.005)
    learner_portvals0 = learner_portvals0 / learner_portvals0.iloc[0]
    cr2, mdr2, sddr2 = stats(learner_portvals0)
    # impact = 0.01
    learner1 = sl.StrategyLearner(verbose=False, impact=0.01)
    learner1.add_evidence(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df1 = learner1.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    learner_portvals1 = msc.compute_portvals(trades_df1, sd, ed, sv, 0, 0.01)
    learner_portvals1 = learner_portvals1 / learner_portvals1.iloc[0]
    cr3, mdr3, sddr3 = stats(learner_portvals1)

    return cr1, cr2, cr3, mdr1, mdr2, mdr3, sddr1, sddr2, sddr3


def plots():
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    commission = 0.00
    impact1=0
    impact2=0.01
    impact3=0.02
    impact4=0.03

    # impact = 0.00
    learner = sl.StrategyLearner(verbose=False, impact=impact1)
    learner.add_evidence(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df = learner.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df = build_trades(trades_df, symbol)
    trades_df['Date'] = pd.to_datetime(trades_df['Date'])
    trades_df.set_index('Date', inplace=True)

    learner_portvals = msc.compute_portvals(trades_df, sd, ed, sv, 0, impact=impact1)
    learner_portvals = learner_portvals / learner_portvals.iloc[0]

    # impact = 0.01
    learner0 = sl.StrategyLearner(verbose=False, impact=impact2)
    learner0.add_evidence(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df0 = learner0.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df0 = build_trades(trades_df0, symbol).assign(Date=lambda df: pd.to_datetime(df['Date'])).set_index('Date')
    learner_portvals0 = msc.compute_portvals(trades_df0, sd, ed, sv, 0, impact=impact2)
    learner_portvals0 = learner_portvals0 / learner_portvals0.iloc[0]

    # impact = 0.02
    learner1 = sl.StrategyLearner(verbose=False, impact=impact3)
    learner1.add_evidence(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df1 = learner1.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df1 = build_trades(trades_df1, symbol).assign(Date=lambda df: pd.to_datetime(df['Date'])).set_index('Date')
    learner_portvals1 = msc.compute_portvals(trades_df1, sd, ed, sv, 0, impact=impact3)
    learner_portvals1 = learner_portvals1 / learner_portvals1.iloc[0]

    # impact = 0.03
    learner2 = sl.StrategyLearner(verbose=False, impact=impact4)
    learner2.add_evidence(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df2 = learner2.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df2 = build_trades(trades_df2, symbol).assign(Date=lambda df: pd.to_datetime(df['Date'])).set_index('Date')
    learner_portvals2 = msc.compute_portvals(trades_df2, sd, ed, sv, 0, impact=impact4)
    learner_portvals2 = learner_portvals2 / learner_portvals2.iloc[0]

    plt.figure(figsize=(18, 9))
    plt.plot(learner_portvals, 'r', label=f'Impact Value = ${impact1}')
    plt.plot(learner_portvals0, 'g', label=f'Impact Value = ${impact2}')
    plt.plot(learner_portvals1, 'b', label=f'Impact Value = ${impact3}')
    plt.plot(learner_portvals2, 'purple', label=f'Impact Value = ${impact4}')
    plt.xticks(rotation=13, fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=20)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Portfolio Value', fontsize=20)
    plt.title('Strategy Learner with Different Impact Values', fontsize=26)
    plt.savefig('Experiment2.png')
    plt.close()

    # initialize datastructures
    avg_returns = []
    number_of_trades = []

    # learner strategy impact1

    trades_df = learner.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df = build_trades(trades_df, symbol).assign(Date=lambda df: pd.to_datetime(df['Date'])).set_index('Date')

    # port stats
    port_value = msc.compute_portvals(trades_df, sd, ed, sv, commission, impact=impact1)
    port_value = port_value / port_value.iloc[0]
    port_value = port_value.to_frame()

    # Update and append data
    avg_returns.append(port_value.pct_change().iloc[:, 0].mean())
    number_of_trades.append(len(trades_df))

    # learner strategy impact2

    trades_df0 = learner0.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df0 = build_trades(trades_df0, symbol).assign(Date=lambda df: pd.to_datetime(df['Date'])).set_index('Date')

    # port stats
    port_value = msc.compute_portvals(trades_df0, sd, ed, sv, commission, impact=impact2)
    port_value = port_value / port_value.iloc[0]
    port_value = port_value.to_frame()

    # Update and append data
    avg_returns.append(port_value.pct_change().iloc[:, 0].mean())
    number_of_trades.append(len(trades_df0))

    # learner strategy impact3

    trades_df1 = learner1.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df1 = build_trades(trades_df1, symbol).assign(Date=lambda df: pd.to_datetime(df['Date'])).set_index('Date')

    # port stats
    port_value = msc.compute_portvals(trades_df1, sd, ed, sv, commission, impact=impact3)
    port_value = port_value / port_value.iloc[0]
    port_value = port_value.to_frame()

    # Update and append data
    avg_returns.append(port_value.pct_change().iloc[:, 0].mean())
    number_of_trades.append(len(trades_df1))

    #learner strategy impact4

    trades_df2 = learner2.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    trades_df2 = build_trades(trades_df2, symbol).assign(Date=lambda df: pd.to_datetime(df['Date'])).set_index('Date')

    #port stats
    port_value = msc.compute_portvals(trades_df2, sd, ed, sv, commission, impact=impact4)
    port_value = port_value / port_value.iloc[0]
    port_value = port_value.to_frame()

    # Update and append data
    avg_returns.append(port_value.pct_change().iloc[:, 0].mean())
    number_of_trades.append(len(trades_df2))

    # pt2 data and charts

    avg = pd.DataFrame(data=np.array(avg_returns),
                       index=[impact1, impact2, impact3, impact4])

    num_t = pd.DataFrame(data=np.array(number_of_trades),
                       index=[impact1, impact2, impact3, impact4])

    # comparison charts
    # figure 1
    fig = plt.figure()
    plt.plot(avg, 'g')
    plt.legend(["Avg_return"])
    plt.xlabel("Impact")
    plt.ylabel("Metric Value")
    plt.title("Average Returns vs Impact")
    plt.savefig("experiment2_avg_rets.png")
    plt.close()

    # Figure 2
    fig = plt.figure()
    plt.plot(num_t, 'r')
    plt.legend(["Number of Trades"])
    plt.xlabel("Impact")
    plt.ylabel("Metric Value")
    plt.title("Number of Trades vs Impact")
    plt.savefig("experiment2_num_trades.png")
    plt.close()

if __name__ == "__main__":
    plots()