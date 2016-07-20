import inspect
import logging
import pandas as pd
import sys; sys.path.append('../..')
from matplotlib.pyplot import show, title

from systems.provided.futures_chapter15.estimatedsystem import futures_system
system=futures_system()
#system.forecastScaleCap.get_scaled_forecast("EDOLLAR", "ewmac64_256").plot()
res=system.rules.get_raw_forecast("CRUDE_W", "carry")
res.to_csv("out.csv")
print (res.head(5))

    
f = '../../sysdata/legacycsv/CRUDE_W_price.csv'
df = pd.read_csv(f,index_col=0,parse_dates=True)
from syscore.accounting import accountCurve
account = accountCurve(df.PRICE, forecast=res)
tmp = account.percent()
print(tmp.stats())

