# F - Jan, G - Feb, H - Mar, J - Apr, K - May, M - Jun
# N - Jul, Q - Aug, U - Sep, V - Oct, W - Nov, Z - Dec
#
import Quandl, os, itertools, sys, logging
import pandas as pd

def download_data():

    fname = '%s/.quandl' % os.environ['HOME']
    if not os.path.isfile(fname):
        print 'Please create a %s file ' % fname
        exit()
    auth = open(fname).read()

    base_dir = "c:/Users/burak/Downloads/futures" 

    years = range(1984,2022)
    months = ['F', 'G', 'H', 'J', 'K', 'M',
              'N', 'Q', 'U', 'V', 'W', 'Z']
    futcsv = pd.read_csv("../data/futures.csv")
    instruments = zip(futcsv.Symbol,futcsv.Market)
    
    for year in years:
        for month in months:
            for (sym,market) in instruments:
                contract = "%s/%s%s%d" % (market,sym,month,year)
                try:
                    print contract
                    df = Quandl.get(contract, returns="pandas",authtoken=auth)
                    print df.head()
                    print df.columns
                except Quandl.Quandl.DatasetNotFound:
                    print "No dataset"
                exit()

                
if __name__ == "__main__":
    
    f = '%(asctime)-15s: %(message)s'
    if len(sys.argv) == 3:
        logging.basicConfig(filename='%s/futures-%d.log' % (os.environ['TEMP'],int(sys.argv[1])),level=logging.DEBUG,format=f)        
        download_data(int(sys.argv[1]),int(sys.argv[2]))
    else:
        logging.basicConfig(filename='%s/futures.log' % os.environ['TEMP'],level=logging.DEBUG, format=f)
        download_data()
