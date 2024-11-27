""""""  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
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
import random
import numpy as np
import pandas as pd
import util

import RTLearner as rt
import BagLearner as bl
from indicators import *
import ManualStrategy as ms
import marketsimcode as msc

np.random.seed(903948218)
random.seed(903948218)
  		  	   		 	   		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	   		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	   		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    # constructor  		  	   		 	   		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose  		  	   		 	   		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		 	   		  		  		    	 		 		   		 		  
        self.commission = commission
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 5}, bags=20, boost=False, verbose=False)

    def author(self):
        return 'pravikumaran3'

    def add_evidence(  		  	   		 	   		  		  		    	 		 		   		 		  
        self,  		  	   		 	   		  		  		    	 		 		   		 		  
        symbol="JPM",
        sd = dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=10000,  		  	   		 	   		  		  		    	 		 		   		 		  
    ):  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		 	   		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		 	   		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		 	   		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  

        syms = [symbol]
        dates = pd.date_range(sd, ed)
        extended_dates = pd.date_range(pd.to_datetime(sd) - pd.Timedelta(days=50), ed)

        # extended prices for indicator data
        extended_prices = get_data([symbol], extended_dates)
        extended_prices = extended_prices[symbol]
        prices_all = util.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices = prices.fillna(method='ffill').fillna(method='bfill')

        # Get all indicators, moving_window = x days
        bbp_data = bollinger_bands(extended_prices, symbol, window=16)
        rsi_data = rsi(extended_prices, symbol, lookback=18)
        macd_data = macd(extended_prices, symbol, short_period=12, long_period=26, signal_period=9)
        # Restrict indicators to the original date range
        bbp_data = bbp_data.loc[sd:ed]
        rsi_data = rsi_data.loc[sd:ed]
        macd_data = macd_data.loc[sd:ed]

        # Constructing trainX
        ind1 = pd.DataFrame(bbp_data).rename(columns={symbol: 'BBP'})
        ind2 = pd.DataFrame(rsi_data).rename(columns={symbol: 'RSI'})
        ind3 = pd.DataFrame(macd_data).rename(columns={symbol: 'MACD'})

        indicators = pd.concat((ind1,ind2,ind3), axis=1)
        indicators.fillna(0, inplace=True)
        indicators = indicators[:-5]
        trainX = indicators.values
        threshold = 0.02
        volatility = prices.pct_change().std()
        if volatility.max() < 0.01:
            threshold = min(threshold, (volatility).max()/2)

        # Constructing trainY
        trainY = []
        for i in range(prices.shape[0] - 5):
            ratio = (prices.iloc[i + 5] - prices.iloc[i]) / prices.iloc[i]
            if ratio.iloc[0] > (threshold + self.impact + self.commission/1000):
                trainY.append(1)
            elif ratio.iloc[0] < (-threshold - self.impact - self.commission/1000):
                trainY.append(-1)
            else:
                trainY.append(0)
        trainY = np.array(trainY)

        # Training
        self.learner.add_evidence(trainX, trainY)

    def testPolicy(  		  	   		 	   		  		  		    	 		 		   		 		  
        self,  		  	   		 	   		  		  		    	 		 		   		 		  
        symbol="JPM",
        sd=dt.datetime(2009, 1, 1),  		  	   		 	   		  		  		    	 		 		   		 		  
        ed=dt.datetime(2010, 12, 31),
        sv=10000,  		  	   		 	   		  		  		    	 		 		   		 		  
    ):  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		 	   		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		 	   		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		 	   		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		 	   		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	   		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	   		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	   		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	   		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		 	   		  		  		    	 		 		   		 		  
        """

        syms = [symbol]
        dates = pd.date_range(sd, ed)
        extended_dates = pd.date_range(pd.to_datetime(sd) - pd.Timedelta(days=50), ed)

        extended_prices = get_data([symbol], extended_dates)
        extended_prices = extended_prices[symbol]
        prices_all = util.get_data(syms, dates)
        prices = prices_all[syms]
        prices = prices.fillna(method='ffill').fillna(method='bfill')

        # Get all indicators, moving_window = x days
        bbp_data = bollinger_bands(extended_prices, symbol, window=16)
        rsi_data = rsi(extended_prices, symbol, lookback=18)
        macd_data = macd(extended_prices, symbol, short_period=12, long_period=26, signal_period=9)
        # Restrict indicators to the original date range
        bbp_data = bbp_data.loc[sd:ed]
        rsi_data = rsi_data.loc[sd:ed]
        macd_data = macd_data.loc[sd:ed]

        # Constructing testX
        ind1 = pd.DataFrame(bbp_data).rename(columns={symbol: 'BBP'})
        ind2 = pd.DataFrame(rsi_data).rename(columns={symbol: 'RSI'})
        ind3 = pd.DataFrame(macd_data).rename(columns={symbol: 'MACD'})

        indicators = pd.concat((ind1, ind2, ind3), axis=1)
        indicators.fillna(0, inplace=True)
        testX = indicators.values
        threshold = 0.02
        volatility = prices.pct_change().std()
        if volatility.max() < 0.01:
            threshold = min(threshold, (volatility).max() / 2)

        # Query learner
        testY = self.learner.query(testX)

        # Constructing trades DataFrame
        trades = prices_all[syms].copy()
        trades.loc[:] = 0
        position = 0
        for i in range(0, prices.shape[0] - 1):
            # Calculate the price ratio considering the market impact
            ratio = (prices.iloc[i + 1] - prices.iloc[i]) / prices.iloc[i]
            if position == 0:
                if ratio.iloc[0] > (threshold + self.impact + self.commission / 1000):  # Adjusted threshold for buying
                    trades.values[i, :] = 1000  # Buy action
                    position = 1
                elif ratio.iloc[0] < (
                        -threshold - self.impact - self.commission / 1000):  # Adjusted threshold for selling
                    trades.values[i, :] = -1000  # Sell action
                    position = -1

            elif position == 1:
                if ratio.iloc[0] < (
                        -threshold - self.impact - self.commission / 1000):  # Adjusted threshold for selling
                    trades.values[i, :] = -2000  # Sell action to reverse position
                    position = -1
                elif ratio.iloc[0] == 0:  # No significant change, sell to neutralize
                    trades.values[i, :] = -1000
                    position = 0

            elif position == -1:
                if ratio.iloc[0] > (threshold + self.impact + self.commission / 1000):  # Adjusted threshold for buying
                    trades.values[i, :] = 2000  # Buy action to reverse position
                    position = 1
                elif ratio.iloc[0] == 0:  # No significant change, buy to neutralize
                    trades.values[i, :] = 1000
                    position = 0

        # Final adjustment based on the last position
        if position == -1:
            trades.values[prices.shape[0] - 1, :] = 1000  # Close position if in sell mode
        elif position == 1:
            trades.values[prices.shape[0] - 1, :] = -1000  # Close position if in buy mode

        return trades

if __name__ == "__main__":
    print("One does not simply think up a strategy")  		  	   		 	   		  		  		    	 		 		   		 		  
