
Given N trading subsystems with a correlation matrix of returns H and
instrument weights W summing to 1, the diversification multiplier will
be 1 ÷ [√(W × H × W T)]. Any negative correlations should be floored
at zero before the calculation is done, to avoid dangerously inflating
the multiplier.



```python
import pandas as pd
df = pd.read_csv('US20_SP500_returns.csv')
print df.tail()
```

```text
        DATETIME      US20     SP500
4866  2016-05-05 -0.000194 -0.000163
4867  2016-05-06 -0.000029  0.000455
4868  2016-05-09  0.000098  0.000058
4869  2016-05-10  0.000000  0.001103
4870  2016-05-11  0.000358 -0.000303
```

```python

```




















