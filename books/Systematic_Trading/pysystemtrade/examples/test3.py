import inspect
import sys; sys.path.append('..')
from sysdata.csvdata import csvFuturesData
from sysdata.configdata import Config
from syscore.algos import robust_vol_calc
import pandas as pd

def calc_ewmac_forecast(price, Lfast, Lslow=None):
    price=price.resample("1B", how="last")
    if Lslow is None: Lslow = 4 * Lfast
    fast_ewma = pd.ewma(price, span=Lfast)
    slow_ewma = pd.ewma(price, span=Lslow)
    raw_ewmac = fast_ewma - slow_ewma
    vol = robust_vol_calc(price.diff())
    return raw_ewmac /  vol

f = '../sysdata/legacycsv/EDOLLAR_price.csv'
price = pd.read_csv(f,index_col=0,parse_dates=True).PRICE
ewmac = calc_ewmac_forecast(price, 32, 128)
ewmac.columns=['forecast']
print(ewmac.tail(5))
ewmac.to_csv('c:/Users/burak/out.csv')

from syscore.accounting import accountCurve
account = accountCurve(price, forecast=ewmac)
tmp = account.percent()
print(tmp.stats())
