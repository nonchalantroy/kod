# F - Jan, G - Feb, H - Mar, J - Apr, K - May, M - Jun
# N - Jul, Q - Aug, U - Sep, V - Oct, W - Nov, Z - Dec
#
import Quandl, os, itertools, sys
from pymongo import MongoClient
import logging, datetime
import pandas as pd
from memo import *

contract_month_codes = ['F', 'G', 'H', 'J', 'K', 'M','N', 'Q', 'U', 'V', 'W', 'Z']

@memo                                    
def get_quandl_auth():
    fname = '%s/.quandl' % os.environ['HOME']
    if not os.path.isfile(fname):
        print 'Please create a %s file ' % fname
        exit()
    auth = open(fname).read()
    return auth

def web_download(contract,start,end):
    df = Quandl.get(contract,trim_start=start,trim_end=end,
                    returns="pandas",authtoken=get_quandl_auth())
    return df

def systemtoday():
    return datetime.datetime.today()

def get(market, sym, month, year, dt, db):
    """
    Returns all data for symbol in a pandas dataframe
    """
    connection = MongoClient()
    db = connection[db]
    q = {"$query" :{"_id": {"sym": sym, "market": market, "month": month,
                            "year": year, "dt": dt }} }
    res = list(db.tickers.find( q ))
    return res

def last_contract(sym, market, db):
    q = { "$query" : {"_id.sym": sym, "_id.market": market} }
    res = db.tickers.find(q).sort([("_id.month",-1), (u"_id.year",-1)]).limit(1)
    return list(res) 

def existing_nonexpired_contracts(sym, market, db, today):
    q = { "$query" : {"_id.sym": sym, "_id.market": market,
                      "_id.month": {"$gte": contract_month_codes[today.month] },
                      "_id.year": {"$gte": today.year } }
    }
    res = db.tickers.find(q)
    return list(res)

def download_data(chunk=1,chunk_size=1,downloader=web_download,
                  today=systemtoday,db="foam",years=(1984,2022)):

    # a tuple of contract years, defining the beginning
    # of time and end of time
    start_year,end_year=years
    months = ['F', 'G', 'H', 'J', 'K', 'M',
              'N', 'Q', 'U', 'V', 'W', 'Z']
    futcsv = pd.read_csv("./data/futures.csv")
    instruments = zip(futcsv.Symbol,futcsv.Market)

    str_start = datetime.datetime(start_year-2, 1, 1).strftime('%Y-%m-%d')
    str_end = today().strftime('%Y-%m-%d')
    today_month,today_year = today().month, today().year
    
    connection = MongoClient()
    tickers = connection[db].tickers

    work_items = []

    # download non-existing / missing contracts - this is the case of
    # running for the first time, or a new contract became available
    # since the last time we ran.
    for (sym,market) in instruments:
        last = last_contract(sym, market, connection[db])
        last_db_year,last_db_month = (0,'A') # init values
        if len(last) > 0: last_year,last_month = last[0]['_id']['year'], last[0]['_id']['month']
        for year in years:
            for month in months:
                if last_db_year < end_year or last_db_month < 'Z':
                    # try to get two years worth of data even for the earliest contract
                    work_items.append([market, sym, month, year])

    # for existing contracts, add to the work queue the download of
    # additional days that are not there. it's a new day, and for for
    # existing non-expired contracts we would have new price data.
    # TBD
    
    for market, sym, month, year in work_items:
        contract = "%s/%s%s%d" % (market,sym,month,year)
        try:
            print contract
            df = downloader(contract,str_start,str_end)
            for srow in df.iterrows():
                dt = str(srow[0])[0:10]
                dt = int(dt.replace("-",""))
                new_row = {"_id": {"sym": sym, "market": market, "month": month, "year": year, "dt": dt },
                           "o": srow[1].Open, "h": srow[1].High,
                           "l": srow[1].Low, "la": srow[1].Last,
                           "s": srow[1].Settle, "v": srow[1].Volume,
                           "oi": srow[1]['Prev. Day Open Interest']
                }

                tickers.save(new_row)

        except Quandl.Quandl.DatasetNotFound:
            print "No dataset"
                    
if __name__ == "__main__":
    
    f = '%(asctime)-15s: %(message)s'
    if len(sys.argv) == 3:
        p = (os.environ['TEMP'],int(sys.argv[1]))
        logging.basicConfig(filename='%s/futures-%d.log' % p,level=logging.DEBUG,format=f)
        download_data(chunk=int(sys.argv[1]),chunk_size=int(sys.argv[2]))
    else:
        logging.basicConfig(filename='%s/futures.log' % os.environ['TEMP'],level=logging.DEBUG, format=f)
        download_data()
