# F - Jan, G - Feb, H - Mar, J - Apr, K - May, M - Jun
# N - Jul, Q - Aug, U - Sep, V - Oct, W - Nov, Z - Dec
#
import Quandl, os, itertools, sys
from pymongo import MongoClient
import logging, datetime
sys.path.append("..")
import pandas as pd
from memo import *

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

def download_data(chunk=1,chunk_size=1,downloader=web_download,
                  today=systemtoday,db="foam"):

    years = range(1984,2022)
    months = ['F', 'G', 'H', 'J', 'K', 'M',
              'N', 'Q', 'U', 'V', 'W', 'Z']
    futcsv = pd.read_csv("../data/futures.csv")
    instruments = zip(futcsv.Symbol,futcsv.Market)

    start="1980-01-01"; end = today().strftime('%Y-%m-%d')
    print start, end

    connection = MongoClient()
    db = connection.foam
    tickers = db.tickers
    
    for year in years:
        for month in months:
            for (sym,market) in instruments:
                contract = "%s/%s%s%d" % (market,sym,month,year)
                try:
                    print contract
                    df = downloader(contract,start,end)
                    for srow in df.iterrows():
                        dt = str(srow[0])[0:10]
                        dt = int(dt.replace("-",""))
                        new_row = {"_id": {"sym": contract, "dt": dt },
                                   "o": srow[1].Open,"h": srow[1].High,
                                   "l": srow[1].Low,"la": srow[1].Last,
                                   "s": srow[1].Settle,"v": srow[1].Volume,
                                   "oi": srow[1]['Prev. Day Open Interest']
                        }
                        tickers.save(new_row)
                    
                except Quandl.Quandl.DatasetNotFound:
                    print "No dataset"
                exit()

                
if __name__ == "__main__":
    
    f = '%(asctime)-15s: %(message)s'
    if len(sys.argv) == 3:
        p = (os.environ['TEMP'],int(sys.argv[1]))
        logging.basicConfig(filename='%s/futures-%d.log' % p,level=logging.DEBUG,format=f)
        download_data(chunk=int(sys.argv[1]),chunk_size=int(sys.argv[2]))
    else:
        logging.basicConfig(filename='%s/futures.log' % os.environ['TEMP'],level=logging.DEBUG, format=f)
        download_data()
