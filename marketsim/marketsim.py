""""""  		  	   		 	   		  		  		    	 		 		   		 		  
"""MC2-P1: Market simulator.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	   		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	   		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	   		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	   		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   		  		  		    	 		 		   		 		  
or edited.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	   		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	   		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Student Name: Pirashan Ravikumaran		  	   		 	   		  		  		    	 		 		   		 		  
GT User ID: pravikumaran3	  	   		 	   		  		  		    	 		 		   		 		  
GT ID: 903948218 		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		 	   		  		  		    	 		 		   		 		  
import os
import numpy as np
import pandas as pd  		  	   		 	   		  		  		    	 		 		   		 		  
from util import get_data, plot_data

def author():
    return 'pravikumaran3'

def studygroup():
    return 'pravikumaran3'
  		  	   		 	   		  		  		    	 		 		   		 		  
def compute_portvals(
    orders_file="./orders/orders.csv",
    start_val=1000000,
    commission=9.95,
    impact=0.005,
):
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param orders_file: Path of the order file or the file object  		  	   		 	   		  		  		    	 		 		   		 		  
    :type orders_file: str or file object  		  	   		 	   		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		 	   		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		 	   		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		 	   		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		 	   		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		 	   		  		  		    	 		 		   		 		  
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input

    #Read the data in from orders file
    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    # datetime format
    orders.index = pd.to_datetime(list(orders.index.values))
    # Sort by dates in descending order
    orders = orders.sort_index()

    # Find earliest and latest dates
    start_date = orders.index.min()
    end_date = orders.index.max()

    # Get unique symbols
    symbols = list(set(orders['Symbol']))

    # orders -> prices -> trades -> holdings -> values -> portfolio value

    # Get prices of symbols
    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices = prices[symbols]
    # Track cash column
    prices['Cash'] = 1.0

    # Trades df with all zeros and changes to stocks/cash
    trades = pd.DataFrame(data=0, columns=prices.columns.values, index=prices.index.values)

    for row in orders.itertuples(index=True):
        symbol = row.Symbol
        shares = row.Shares
        order_type = row.Order
        date = row.Index

        if order_type == "BUY":
            trades.at[date, symbol] += shares
            # Calculate cash for the trade
            cash_spent = -1 * prices.at[date, symbol] * shares
            # Update cash in trades
            trades.at[date, "Cash"] += cash_spent - (impact * abs(cash_spent)) - commission

        else:
            trades.at[date, symbol] -= shares
            # Calculate cash for the trade
            cash_earned = prices.at[date, symbol] * shares
            # Update cash in trades
            trades.at[date, "Cash"] += cash_earned - (impact * cash_earned) - commission

    # holdings df
    holdings= pd.DataFrame(data=0, columns=trades.columns.values, index=trades.index.values)
    holdings.iloc[[0]] = trades.iloc[[0]]
    holdings.Cash.iat[0] += start_val

    for i in range(1, holdings.shape[0]):
        holdings.loc[holdings.index[i]] = holdings.loc[holdings.index[i - 1]] + trades.loc[holdings.index[i]]

    values = prices * holdings
    portvals = values.sum(axis=1)

    return portvals
  		  	   		 	   		  		  		    	 		 		   		 		  
def test_code():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Helper function to test code  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		  	   		 	   		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		  	   		 	   		  		  		    	 		 		   		 		  
    # Define input parameters  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    of = "./orders/orders-01.csv"
    sv = 1000000  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Process orders  		  	   		 	   		  		  		    	 		 		   		 		  
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		 	   		  		  		    	 		 		   		 		  
    if isinstance(portvals, pd.DataFrame):  		  	   		 	   		  		  		    	 		 		   		 		  
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		 	   		  		  		    	 		 		   		 		  
    else:  		  	   		 	   		  		  		    	 		 		   		 		  
        "warning, code did not return a DataFrame"  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Get portfolio stats  		  	   		 	   		  		  		    	 		 		   		 		  
    # Here we just fake the data. you should use your code from previous assignments.  		  	   		 	   		  		  		    	 		 		   		 		  
    start_date = portvals.index[0]
    end_date = portvals.index[-1]
    daily_ret = (portvals / portvals.shift(1)) - 1
    daily_ret.iloc[0] = 0
    daily_ret = daily_ret[1:]
    cum_ret = (portvals[-1]/portvals[0]) - 1
    avg_daily_ret = daily_ret.mean()
    std_daily_ret = daily_ret.std()
    sharpe_ratio = avg_daily_ret / std_daily_ret * np.sqrt(252)

    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [  		  	   		 	   		  		  		    	 		 		   		 		  
        0.2,  		  	   		 	   		  		  		    	 		 		   		 		  
        0.01,  		  	   		 	   		  		  		    	 		 		   		 		  
        0.02,  		  	   		 	   		  		  		    	 		 		   		 		  
        1.5,  		  	   		 	   		  		  		    	 		 		   		 		  
    ]  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Compare portfolio against $SPX  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Date Range: {start_date} to {end_date}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print()  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print()  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print()  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print()  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print()  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    test_code()  		  	   		 	   		  		  		    	 		 		   		 		  
