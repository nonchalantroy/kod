import sys; sys.path.append('..')
from syscore.algos import robust_vol_calc
import pandas as pd
from syscore.dateutils import BUSINESS_DAYS_IN_YEAR, ROOT_BDAYS_INYEAR, WEEKS_IN_YEAR, ROOT_WEEKS_IN_YEAR
from syscore.dateutils import MONTHS_IN_YEAR, ROOT_MONTHS_IN_YEAR

DEFAULT_CAPITAL = 10000000.0
DEFAULT_ANN_RISK_TARGET = 0.16
DEFAULT_DAILY_CAPITAL=DEFAULT_CAPITAL * DEFAULT_ANN_RISK_TARGET / ROOT_BDAYS_INYEAR

def sharpe(price, forecast):        
    base_capital = DEFAULT_CAPITAL
    daily_risk_capital = DEFAULT_CAPITAL * DEFAULT_ANN_RISK_TARGET / ROOT_BDAYS_INYEAR        
    ts_capital=pd.Series([DEFAULT_CAPITAL]*len(price), index=price.index)        
    ann_risk = ts_capital * DEFAULT_ANN_RISK_TARGET
    get_daily_returns_volatility = robust_vol_calc(price.diff())
    multiplier = daily_risk_capital * 1.0 * 1.0 / 10.0
    denominator = get_daily_returns_volatility
    numerator = forecast *  multiplier
    positions = numerator.ffill() /  denominator.ffill()
    use_positions = positions.shift(1)
    cum_trades = use_positions.ffill()
    trades_to_use=cum_trades.diff()
    price_returns = price.diff()
    instr_ccy_returns = cum_trades.shift(1)* price_returns 
    instr_ccy_returns=instr_ccy_returns.cumsum().ffill().reindex(price.index).diff()
    base_ccy_returns = instr_ccy_returns 
    mean_return = base_ccy_returns.mean() * BUSINESS_DAYS_IN_YEAR
    vol = base_ccy_returns.std() * ROOT_BDAYS_INYEAR
    return mean_return / vol

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

from syscore.accounting2 import sharpe
print (sharpe(df.PRICE, forecast))
