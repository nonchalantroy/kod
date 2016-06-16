import sys; sys.path.append('..')
from sysdata.csvdata import csvFuturesData
from sysdata.configdata import Config
from syscore.algos import robust_vol_calc
import pandas as pd

def calc_ewmac_forecast(price, Lfast, Lslow=None):
    price=price.resample("1B", how="last")
    fast_ewma = pd.ewma(price, span=Lfast)
    slow_ewma = pd.ewma(price, span=Lslow)
    raw_ewmac = fast_ewma - slow_ewma
    vol = robust_vol_calc(price.diff())
    return raw_ewmac /  vol

f = '../sysdata/legacycsv/EDOLLAR_price.csv'
df = pd.read_csv(f,index_col=0,parse_dates=True)

fast_ewma = pd.ewma(df.PRICE, span=32)
slow_ewma = pd.ewma(df.PRICE, span=128)
raw_ewmac = fast_ewma - slow_ewma
vol = robust_vol_calc(df['PRICE'].diff())
forecast = raw_ewmac /  vol 
positions = forecast / vol
print (positions.tail(4))

print (positions.diff().mean())
print (vol.mean())
print (positions.diff().mean() / vol.mean())

# sharpe 0.50

