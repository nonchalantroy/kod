# F - Jan, G - Feb, H - Mar, J - Apr, K - May, M - Jun
# N - Jul, Q - Aug, U - Sep, V - Oct, W - Nov, Z - Dec
#
import Quandl, os, itertools

fname = '%s/.quandl' % os.environ['HOME']
if not os.path.isfile(fname):
    print 'Please create a %s file ' % fname
    exit()
auth = open(fname).read()

base_dir = "c:/Users/burak/Downloads/futures" 

years = range(1984,2022)
months = ['F', 'G', 'H', 'J', 'K', 'M',
          'N', 'Q', 'U', 'V', 'W', 'Z']
#instruments = ['CME/CL'] # oil
#instruments = ['CME/KC'] # coffee
#instruments = ['CME/TY'] # US-10 treasury
instruments = ['CME/ED'] # eurodollar

for year in years:
    for month in months:
        for code in instruments:
            file = "%s%s%d" % (code,month,year)
            fout = base_dir + "/%s.csv" % file.replace("/","-")
            print file
            if os.path.isfile(fout):
                print "file exists, skipping..."
                continue
            try:
                df = Quandl.get(file, returns="pandas",authtoken=auth)
            except Quandl.Quandl.DatasetNotFound:
                print "No dataset"
                continue
            print fout
            df.to_csv(fout)

            
