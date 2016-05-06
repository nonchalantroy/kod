from urllib import urlretrieve
from pymongo import MongoClient
import numpy as np, pandas as pd, sys
import datetime as dt, requests
import datetime, re, StringIO

def download(symbol, days=11):
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
    
    df['Timestamp2'] = df.Timestamp.map(lambda x: x.strftime('%Y%m%d'))
    df['Time'] = df.Timestamp.map(lambda x: int(x.strftime('%H%M%S')))
    return  df

def insert_hft(df, symbol):
    connection = MongoClient()
    db = connection.foam    
    for dt in df.Timestamp2.unique():
        tmp = df[df.Timestamp2 == dt]
        tmp = tmp.drop(['Timestamp2','Timestamp'],axis=1)
        tmp.Time = tmp.Time.astype(str)
        tmp = tmp.set_index('Time')
        tmp = tmp.to_dict(orient='index')
        q = {"$query" :{"_id.sym": symbol, "_id.dt": int(dt)} }
        res = list(db.tickers.find(q).limit(1))
        if len(res) > 0:
            res = res[0]
            res['hft'] = tmp
            db.tickers.save(res)
    
if __name__ == "__main__":
    # hft data
    hft_symbols = pd.read_csv("./data/hft.dat")

    for symbol in list(hft_symbols.Symbols):
        print 'HFT download for ' + symbol
        dh = download(symbol)
        insert_hft(dh, symbol)
