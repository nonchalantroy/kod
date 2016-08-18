import inspect
import logging
import pandas as pd
import sys; sys.path.append('../..')
from matplotlib.pyplot import show, title

def carry(daily_ann_roll, vol, smooth_days=90):
    ann_stdev = vol * 256
    raw_carry = daily_ann_roll / ann_stdev
    smooth_carry = pd.ewma(raw_carry, smooth_days)
    return smooth_carry

from syscore.algos import robust_vol_calc

f = '../../sysdata/legacycsv/MXP_price.csv'
df = pd.read_csv(f,index_col=0,parse_dates=True)
vol = robust_vol_calc(df.PRICE.diff())
f = '../../sysdata/legacycsv/MXP_carrydata.csv'
df2 = pd.read_csv(f,index_col=0,parse_dates=True)
carryoffset = 3
forecast2 = carry(-1*(df2.CARRY-df2.PRICE)/(carryoffset/12.), vol) * 30.
from syscore.accounting import accountCurve
account = accountCurve(df.PRICE, forecast=forecast2)
print (account.sharpe())
