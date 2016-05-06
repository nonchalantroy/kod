

```python
import sys; sys.path.append('..')
import foam
df = foam.get_multi(['SBUX'])
df1 = df[(df.index > '2008-01-01') & (df.index <= '2016-01-01')]
#df1 = df[(df.index > '2008-01-01') ]
df1.plot()
plt.savefig('simple_01.png')
```

```python
import sys; sys.path.append('..')
import foam
df = foam.get_multi(['GOOG'])
df = df[df.index > '2015-01-01']
df.plot()
plt.savefig('simple_02.png')
```


```python
import sys; sys.path.append('..'); sys.path.append('../quant_at')
import foam
import pyconometrics

df = foam.get_multi(['XLE','USO'])
#df = df.ix[(df.index>'2009-01-01') & (df.index<'2012-01-01')]
df = df.ix[(df.index>'2009-01-01') & (df.index<'2011-01-01')]
print '%99,%95,%90'
print pyconometrics.cadf(np.matrix(df['XLE']).H,
                         np.matrix(df['USO']).H,0,1)

```

```text %99,%95,%90 {'adf': -2.6299960115993088, 'alpha':
-0.026636401070966147, 'nlag': 1, 'crit': matrix([[-3.88031, -3.35851,
-3.03798, -1.01144, -0.65334, 0.15312]]), 'nvar': 1} ```


```python
import sys; sys.path.append('../quant_at')
import sys; sys.path.append('../quant_at')
import foam
import pyconometrics

df = foam.get_multi(['EWA','EWC'])
#df = df.ix[(df.index>'2007-01-01') & (df.index<'2012-01-01')]
df = df.ix[(df.index>'2010-01-01') & (df.index<'2016-01-01')]
print '%99,%95,%90'
print pyconometrics.cadf(np.matrix(df['EWA']).H,
                         np.matrix(df['EWC']).H,0,1)

```

```text %99,%95,%90 {'adf': -1.8886920141331247, 'alpha':
-0.006621093613177381, 'nlag': 1, 'crit': matrix([[-3.88031, -3.35851,
-3.03798, -1.01144, -0.65334, 0.15312]]), 'nvar': 1} ```













