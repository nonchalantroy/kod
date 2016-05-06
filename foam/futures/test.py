import Quandl, os, itertools

fname = '%s/.quandl' % os.environ['HOME']
if not os.path.isfile(fname):
    print 'Please create a %s file ' % fname
    exit()
auth = open(fname).read()

base_dir = "c:/Users/burak/Downloads/futures" 

years = range(1984,2022)
months = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'W', 'Z']
#instruments = [('oil','CL')]
instruments = [('coffee','KC')]

for year in years:
    for month in months:
        for (ins,code) in instruments:
            file = "CME/%s%s%d" % (code,month,year)
            fout = base_dir + "/%s/%s.csv" % (ins,file)
            fout = fout.replace("CME/","CME-")
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
