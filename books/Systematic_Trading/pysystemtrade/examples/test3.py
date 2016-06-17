import inspect
import inspect
import sys; sys.path.append('..')
from syscore.algos import robust_vol_calc
import pandas as pd

def calc_ewmac_forecast(price, Lfast, Lslow):
    price=price.resample("1B", how="last")
    fast_ewma = pd.ewma(price, span=Lfast)
    slow_ewma = pd.ewma(price, span=Lslow)
    raw_ewmac = fast_ewma - slow_ewma
    vol = robust_vol_calc(price.diff())
    return raw_ewmac /  vol

#f = '../sysdata/legacycsv/EDOLLAR_price.csv'
f = '../sysdata/legacycsv/CORN_price.csv'
price = pd.read_csv(f,index_col=0,parse_dates=True).PRICE
ewmac = calc_ewmac_forecast(price, 32, 128)
ewmac.columns=['forecast']
print(ewmac.tail(5))

from syscore.accounting import accountCurve
account = accountCurve(price, forecast=ewmac)
tmp = account.percent()
print(tmp.stats())
