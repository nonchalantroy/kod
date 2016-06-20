
with zipfile.ZipFile('c:/Users/burak/Documents/classnotes/tser/tser_voltar/legacycsv.zip', 'r') as z:
    df =  pd.read_csv(z.open('SP500_price.csv'),sep=',',index_col=0,parse_dates=True)
    df['NASDAQ'] =  pd.read_csv(z.open('NASDAQ_price.csv'),sep=',',index_col=0,parse_dates=True)
    df['US20'] =  pd.read_csv(z.open('US20_price.csv'),sep=',',index_col=0,parse_dates=True)

df.columns = ['SP500','NASDAQ','US20']

df['SP500'] = df.SP500.pct_change()
df['NASDAQ'] = df.NASDAQ.pct_change()
df['US20'] = df.US20.pct_change()

df = df[(df.index >= '1999-08-02') & (df.index <= '2015-04-22')]
    
random.seed(0)

mat1=optimise_over_periods(df)
mat1.plot()
plt.show()
