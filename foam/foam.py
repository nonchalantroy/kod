from pymongo import MongoClient
import logging, Quandl, random, os, hft
import datetime, glob, pandas as pd
from pandas_datareader import data, wb
import numpy as np, sys
from memo import *

@memo                                    
def get_quandl_auth():
    f = '.quandl'
    if not os.path.isfile(f): print '%s must exist\n' % f; exit(1)
    return open(f).read()

def web_load(symbol, backend, start, end):
    """
    Outside interface to get all the data
    """
    auth = get_quandl_auth()
    try: 
        if backend == "fred":
            return data.DataReader(symbol, backend, start, end)
        if backend == "google":
            return data.DataReader(market + ":" + symbol, backend, start, end)
        if backend == "yahoo":
            return data.DataReader(symbol, backend, start, end)
    except IOError:
        logging.debug("cant find " + symbol)

def get_beginning_of_time():
    return datetime.datetime(2006, 1, 1)

def get_today():
    today=datetime.datetime(2016, 2, 15) # hack, freeze the end time
    #today=datetime.datetime.today() - datetime.timedelta(days=1)
    today_int = int(today.strftime('%Y%m%d') )
    return today, today_int

def get_last_date_in_db(symbol, db, today):
    ts = db.tickers.find( {"_id.sym": symbol} )
    # Check if there are records.
    if ts.count() > 0:
        q = {"$query" :{"_id.sym": symbol},"$orderby":{"_id.dt" : -1}}
        ts = list(db.tickers.find(q).limit(1))
        last_date_in_db = int(ts[0]['_id']['dt'])        
        return pd.to_datetime(str(last_date_in_db), format='%Y%m%d')    
    
def do_download(items):
    """
    Download a given list of (market,symbol,name) triplets.
    This list would have been prepared outside of this call, probably
    a chunk of bigger list of symbols. This way this function has no
    knowledge of what all symbols are, it only works on the piece given
    to it.
    """

    connection = MongoClient()
    db = connection.foam
    tickers = db.tickers

    beginning_of_time=get_beginning_of_time()
    today, today_int = get_today()

    backend = "yahoo"
    #backend = "google"

    logging.debug ("%d items" % len(items))
    for market,symbol,name in items:
        
        logging.debug("%s %s" % (symbol, name))
        s = None; last_date_in_db = None
        if market == "fred":
            last_date = get_last_date_in_db(symbol, db, today)
            logging.debug('last %s' % last_date)
            logging.debug('today %s' % today)
            if last_date and last_date >= today:
                logging.debug('no need')
                continue            
            s = web_load(symbol, "fred", beginning_of_time, today)
            if 'DataFrame' not in str(type(s)): continue
            
            for srow in s.iterrows():
                dt = str(srow[0])[0:10]
                dt = int(dt.replace("-",""))
                new_row = {"_id": {"sym": symbol, "dt": dt },"a": float(srow[1])}
                tickers.save(new_row)

        elif market == "nasdaq" or market == "nyse" or market =="amex" or market == "etfs" :
            start = beginning_of_time; end = today            
            last_date = get_last_date_in_db(symbol,db,today)
            logging.debug('last %s' % last_date)
            logging.debug('today %s' % today)            
            if last_date and last_date >= today:
                logging.debug('no need')
                continue
                
            if last_date: start = last_date            
            logging.debug("" + repr(start) + " " + repr(end))

            s = web_load(symbol, backend, start, end)

            # symbol could not be found
            if 'DataFrame' not in str(type(s)): continue
                    
            for srow in s.iterrows():
                dt = int((str(srow[0])[0:10]).replace("-",""))
                new_row = {"_id": {"sym": symbol, "dt": dt },
                           "o": srow[1].Open,
                           "h": srow[1].High,
                           "l": srow[1].Low,
                           "c": srow[1].Close,
                           "v": srow[1].Volume,
                           "a": srow[1]['Adj Close']}

                tickers.save(new_row)

                
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]
        
def download_data(ith_chunk=0, no_chunks=1,base_dir="."):
    """
    Download data for the ith chunk of no_chunks. The chunks
    come from a list of all available stock, etf symbols
    """
    res = []
    for f in glob.glob(base_dir + '/data/*.csv'):
        market = f.replace(base_dir + "/data/","").replace(".csv","")
        df = pd.read_csv(f)
        for line in df.iterrows():
            res.append((market, line[1].Symbol, line[1].Name))


    random.seed(0)
    random.shuffle(res)
    
    s =  int(len(res) / no_chunks)    
    res = list(chunks(res, s))    
    do_download(res[ith_chunk])

def get(symbol):
    """
    Returns all data for symbol in a pandas dataframe
    """
    connection = MongoClient()
    db = connection.foam
    
    q = {"$query" :{"_id.sym": symbol},"$orderby":{"_id.dt" : 1}}
    res = list(db.tickers.find( q )); res1 = []
    if len(res) == 0: return pd.DataFrame()
    if 'c' in res[0]: # then we have a stock ticker, this series does not have 'closed' or 'open'
        for x in res: res1.append( { 'a': x['a'],'c': x['c'],'h':x['h'], \
                                   'l': x['l'],'o': x['o'],'Date':x['_id']['dt']} )
    else: # we have a macro timeseries, 'a' always exists in all time series
        for x in res: res1.append( { 'a': x['a'],'Date':x['_id']['dt']} )
            
    df = pd.DataFrame(res1, columns=res1[0].keys())
    df['Date'] = pd.to_datetime(df.Date,format='%Y%m%d')
    df = df.set_index('Date')    
    return df

def get_multi(symbols):
    """
    Returns all data for symbols
    """
    dfs = [get(x).a for x in symbols]
    res = pd.concat(dfs,axis=1)
    res.columns = symbols
    return res

def get_hft(symbol, date):
    """
    Return minute level high-frequency data for the given symbol and date
    """
    connection = MongoClient()
    db = connection.foam
    q = {"$query" :{"_id.sym": symbol, "_id.dt": date} }
    res = list(db.tickers.find(q).limit(1))
    if len(res) > 0 and 'hft' in res[0].keys():
        df = pd.DataFrame(res[0]['hft'])
        return df.T

def get_hft_for_dates(symbol, start, end):
    """
    Return minute level high-frequency data for a time period
    between start and end.
    """
    start = pd.to_datetime(str(start), format='%Y%m%d')
    end = pd.to_datetime(str(end), format='%Y%m%d')
    dates = [(start+datetime.timedelta(days=i)).strftime('%Y%m%d') for i in range((end-start).days+1)]
    res = []
    for dt in dates:
        df = get_hft(symbol, int(dt))
        if 'DataFrame' in str(type(df)):
            df['Date'] = dt
            df = df.set_index('Date',append=True)
            res.append(df)
    dfs = pd.concat(res)
    return dfs

        
    
if __name__ == "__main__":
    
    f = '%(asctime)-15s: %(message)s'
    if len(sys.argv) == 3:
        logging.basicConfig(filename='/tmp/stocks-%d.log' % int(sys.argv[1]),level=logging.DEBUG,format=f)        
        download_data(int(sys.argv[1]),int(sys.argv[2]))
    else:
        logging.basicConfig(filename='/tmp/stocks.log',level=logging.DEBUG, format=f)
        download_data()
        
