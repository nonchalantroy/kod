from urllib import urlretrieve
import numpy as np, pandas as pd, sys
import datetime as dt, requests
import datetime, re, StringIO

if len(sys.argv) < 3: print "Usage: test_hf.py [symbol] [lookback, in number of days]"; exit(1)

symbol = sys.argv[1]
days = int(sys.argv[2])

url='http://chartapi.finance.yahoo.com/instrument/1.0/%s/chartdata;type=quote;range=%dd/csv' % (symbol,days)
response = requests.get(url)
response_body = response.content
content = StringIO.StringIO(response_body)

res = []
for x in content.readlines():
    if ":" in x: continue
    res.append(x.strip().split(','))

df = pd.DataFrame(res, columns=['Timestamp','close','high','low','open','volume'])
df.Timestamp = df.Timestamp.map(lambda x: datetime.datetime.fromtimestamp(float(x)))

df.to_csv("/tmp/%s-%dd.csv" % (symbol,days),index=None)

