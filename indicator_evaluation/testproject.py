import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data
from indicators import bollinger_bands, rsi, stochastic_indicator, macd, cci
import TheoreticallyOptimalStrategy as tos
from marketsimcode import compute_portvals


def author():
    return 'pravikumaran3'

def studygroup():
    return 'pravikumaran3'

if __name__ == "__main__":
    # parameters for stock data
    start_date = dt.date(2008, 1, 1)
    end_date = dt.date(2009, 12, 31)
    date_ranges = pd.date_range(start_date, end_date)
    symbol = 'JPM'

    prices = get_data([symbol], date_ranges)
    high_prices = get_data([symbol], date_ranges, colname="High")
    low_prices = get_data([symbol], date_ranges, colname="Low")
    close_prices = get_data([symbol], date_ranges, colname="Close")
    prices = prices.drop(columns="SPY")

    # TOS
    trades = tos.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)

    # recreate the orders file input needed for marketsimcode
    orders = pd.DataFrame(index=trades.index.values, columns=["Symbol", "Order", "Shares"])
    orders["Symbol"] = "JPM"
    for idx, row in trades.iterrows():
        if row['Trades'] > 0:
            orders.at[idx, 'Order'] = 'BUY'
        elif row['Trades'] < 0:
            orders.at[idx, 'Order'] = 'SELL'
    orders["Shares"] = abs(trades)

    # portfolio normalized
    port_value = compute_portvals(orders, start_date, end_date)
    port_value = port_value / port_value.iloc[0]

    # benchmark orders df
    bench_orders = pd.DataFrame(index=trades.index.values, columns=['Symbol', 'Order', 'Shares'])
    bench_orders.iloc[0] = ['JPM', 'BUY', 1000]
    bench_orders.iloc[1:] = [['JPM', 'HOLD', 0] for date in trades.index[1:]]
    # benchmark normalized
    bench_value = compute_portvals(bench_orders, start_date, end_date, start_val=100000)
    bench_value = bench_value / bench_value.iloc[0]

    # Create Chart
    fig = plt.figure()
    plt.plot(bench_value, color='purple', label='Benchmark', linewidth=2)
    # note legend goes after plot
    plt.plot(port_value, color='red', label='Optimized Portfolio', linewidth=2)
    plt.legend(["Benchmark", "Optimized Portfolio"])
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("TOS Portfolio vs Benchmark Portfolio")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid()
    # plt.text(0.5, 0.5, "pravikumaran3@gatech.edu", fontsize=30, rotation=50, color='gray',
    #          alpha=0.5, ha='center', va='center', transform=fig.transFigure)
    plt.savefig("TOS_Portfolio_vs_Benchmark.png")
    plt.close()

    # Create Table
    # Stats for Daily Rets
    daily_rets_port = port_value.pct_change().dropna()
    daily_rets_bench = bench_value.pct_change().dropna()
    # Bench
    bench_std = round(daily_rets_bench.std(), 6)
    bench_cum_rets = round((bench_value[-1] / bench_value[0]) - 1, 6)
    bench_avg_rets = round(daily_rets_bench.mean(), 6)
    # portfolio
    port_std = round(daily_rets_port.std(), 6)
    port_cum_rets = round((port_value[-1] / port_value[0]) - 1, 6)
    port_avg_rets = round(daily_rets_port.mean(), 6)

    # Table File
    table_data = {
        'Portfolio': ['Benchmark', 'TOS Portfolio'],
        'STD': [f"{bench_std:.6f}", f"{port_std:.6f}"],
        'Cumulative Returns': [f"{bench_cum_rets:.6f}", f"{port_cum_rets:.6f}"],
        'Average Returns': [f"{bench_avg_rets:.6f}", f"{port_avg_rets:.6f}"]
    }

    # Create DataFrame from dictionary
    df = pd.DataFrame(table_data)

    # Write to a CSV file
    output_filename = 'p6_results.txt'
    df.to_csv(output_filename, header=True, index=False, sep='\t', mode='w')

    # Bollinger Bands
    bb_percent = bollinger_bands(prices, symbol)

    # RSI
    data = rsi(prices, symbol, lookback=14)

    # Stochastic
    data = stochastic_indicator(prices, symbol, lookback_k=14, lookback_d=3)

    # MACD
    data = macd(prices, symbol, short_period=12, long_period=26, signal_period=9)

    # CCI
    data = cci(prices, symbol, lookback=20, high_prices=high_prices, low_prices=low_prices, close_prices=close_prices)