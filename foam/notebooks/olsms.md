
```python
import sys, pandas as pd
sys.path.append('..'); import foam
df = foam.get_multi(['MCD','CMG','IBB'])
df = df.ix[(df.index > '2010-01-01') & (df.index < '2011-01-01')]
```

```python
import statsmodels.api as sm
lookback = 100
x = np.ones((lookback,2))
for t in range(lookback,lookback+20):
    x[:,1] = np.array(range(lookback))
    y = df2.MCD[t-lookback:t]
    f = sm.OLS(y,x).fit()
    #print t, f.params[0], f.params[1]
    df2.loc[df2.index[t],'intercept'] = f.params[0]
    df2.loc[df2.index[t],'slope'] = f.params[1]
```


```python
df2.MCD.plot()
plt.savefig('simple_01.png')
```


```python
df2['RM'] = pd.rolling_mean(df2.CMG, window=20)
df2[['CMG','RM']].plot()
plt.savefig('simple_03.png')
```
































