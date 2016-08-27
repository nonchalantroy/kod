import urllib2, datetime, pickle
import pandas as pd

p = open('../data.pkl', 'rb')
tmp = pickle.load(p)
print len(tmp)
exit()

start_date = pd.to_datetime("20000101", format='%Y%m%d')
end_date = pd.to_datetime("20160101", format='%Y%m%d')
delta = end_date - start_date
dates = []
for i in range(delta.days + 1): dates.append(start_date + datetime.timedelta(days=i))    

news = {}
for i,d in enumerate(dates):
    d = d.strftime('%Y/%m/%d')
    print d
    url = "http://www.milliyet.com.tr/%s/haber/index.html" % d
    html1 = urllib2.urlopen(url).read()
    url = "http://www.milliyet.com.tr/%s/index.html" % d
    html2 = urllib2.urlopen(url).read()
    news[d] = html1 + html2
    if i == 30: break

output = open('../data.pkl', 'wb')
pickle.dump(news, output)
output.close()    
