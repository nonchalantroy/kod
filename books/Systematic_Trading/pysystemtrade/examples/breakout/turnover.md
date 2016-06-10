
```python
import pandas as pd
df = pd.read_csv('out.csv',index_col=0,parse_dates=True)
df.columns=['price']
```

```python
#avg =  (df.price / 10).diff().abs().resample("1B", how="sum").mean()
avg =  (df.price / 10).diff().abs().mean()
print avg * 256
```

```text
28.7998781255
```










