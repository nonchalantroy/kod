import inspect
import sys; sys.path.append('../..')
from syscore.accounting import account_test
from syscore.pdutils import turnover
from sysdata.configdata import Config
from systems.provided.futures_chapter15.estimatedsystem import futures_system
from systems.provided.moretradingrules.morerules import breakout
import pandas as pd
import numpy as np
from matplotlib.pyplot import show, legend, matshow
from syscore.dateutils import BUSINESS_DAYS_IN_YEAR


bvariations=["breakout"+str(ws) for ws in [10, 20, 40, 80, 160, 320]]
evariations=["ewmac%d_%d" % (fast, fast*4) for fast in [2,4,8,16,32, 64]]

my_config = Config("examples.breakout.breakoutfuturesestimateconfig.yaml")
system = futures_system(config=my_config, log_level="on")
price=system.data.daily_prices("CRUDE_W")
lookback=250
roll_max = pd.rolling_max(price, lookback, min_periods=min(len(price), np.ceil(lookback/2.0)))
roll_min = pd.rolling_min(price, lookback, min_periods=min(len(price), np.ceil(lookback/2.0)))
all=pd.concat([price, roll_max, roll_min], axis=1)
all.columns=["price", "max",  "min"]

roll_mean = (roll_max+roll_min)/2.0
all=pd.concat([price, roll_max, roll_mean,roll_min], axis=1)
all.columns=["price", "max", "mean", "min"]

def to(x, y):
    """
    Gives the turnover of x, once normalised for y    
    Returned in annualised terms
    """    
    if type(y) is float:
        y=pd.Series([y]*len(x.index) , x.index)    
    norm_x= x / y.ffill()    
    avg_daily=float(norm_x.diff().abs().resample("1B", how="sum").mean())
    return avg_daily*BUSINESS_DAYS_IN_YEAR

## gives a nice natural scaling
output = 40.0*((price - roll_mean) / (roll_max - roll_min))
output.to_csv('out.csv')
print(to(output, 10.0))
