import inspect
import sys; sys.path.append('..')
from sysdata.csvdata import csvFuturesData
from sysdata.configdata import Config
from syscore.algos import robust_vol_calc
import pandas as pd

def calc_ewmac_forecast(price, Lfast, Lslow=None):
    """
    Calculate the ewmac trading fule forecast, given a price and EWMA speeds Lfast, Lslow and vol_lookback

    """
    # price: This is the stitched price series
    # We can't use the price of the contract we're trading, or the volatility will be jumpy
    # And we'll miss out on the rolldown. See
    # http://qoppac.blogspot.co.uk/2015/05/systems-building-futures-rolling.html

    price=price.resample("1B", how="last")

    if Lslow is None:
        Lslow = 4 * Lfast

    # We don't need to calculate the decay parameter, just use the span
    # directly

    fast_ewma = pd.ewma(price, span=Lfast)
    slow_ewma = pd.ewma(price, span=Lslow)
    raw_ewmac = fast_ewma - slow_ewma

    vol = robust_vol_calc(price.diff())

    return raw_ewmac /  vol

df = pd.read_csv('../sysdata/legacycsv/EDOLLAR_price.csv',index_col=0,parse_dates=True)

price = df.PRICE
ewmac = calc_ewmac_forecast(price, 32, 128)
ewmac.columns=['forecast']
print(ewmac.tail(5))
ewmac.to_csv('c:/Users/burak/out.csv')

from syscore.accounting import accountCurve
account = accountCurve(price, forecast=ewmac)
tmp = account.percent()
print(tmp.stats())
