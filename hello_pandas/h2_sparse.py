import pandas as pd
from typing import List, Tuple

d = pd.DataFrame([
    pd.Series({0: 1, 10: 1}),
    pd.Series({1: 1, 6: 1}),
    pd.Series({3: 1, 4: 1, 5: 1, 7: 1}),
], )

print(d)

f = open("h2.csv", "w")
f.write(d.to_csv(index=True))
f.close()
print('ngon')