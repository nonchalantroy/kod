import pandas as pd, datetime
from pandas_datareader import data

# Stocks
start=datetime.datetime(2013, 1, 1)
end=datetime.datetime(2015, 9, 30)
s = data.DataReader("SSW", 'yahoo', start, end)
print s.head()

# Options

from pandas.io.data import Options

aapl = Options('AAPL',"yahoo")
df = aapl.get_options_data()
print df.head()
