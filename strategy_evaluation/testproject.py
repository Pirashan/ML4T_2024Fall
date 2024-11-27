import random

import ManualStrategy
import StrategyLearner as sl
import experiment1
import experiment2
from marketsimcode import compute_portvals
from util import get_data

import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np

np.random.seed(903948218)

def author():
    return "pravikumaran3"

def results():
    # Create an instance of the ManualStrategy class
    manual_strategy = ManualStrategy.ManualStrategy(symbol="JPM", verbose=True)
    # Test and print in-sample performance
    manual_strategy.plot_in_sample()
    # Test and print out-of-sample performance
    manual_strategy.plot_out_of_sample()
    experiment1.plots()
    experiment2.plots()

if __name__ == "__main__":
    results()