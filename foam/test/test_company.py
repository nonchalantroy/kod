import glob, pandas as pd
from yahoo_finance import Share

res = []
for f in glob.glob('data/*.csv'):
    market = f.replace("data/","").replace(".csv","")
    df = pd.read_csv(f)
    for line in df.iterrows():
        res.append((market, line[1].Symbol, line[1].Name))

for (market,symbol,name) in res:
    if market=='nyse':
        x = Share(symbol)
        print x.get_book_value()
        print x.get_ebitda()
        print x.get_earnings_share()
        print x.get_price_sales()
        
