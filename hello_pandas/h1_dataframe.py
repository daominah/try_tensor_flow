import pandas as pd
import numpy as np

d = pd.DataFrame(
    np.random.randn(6, 4),
    index=['zero', 'one', 'two', 'three', 'four', 'five'],
    columns=list('ABCD'))

print(d, '\n')
print("d['B'][0]: ", d['B'][3], d['B']['three'])
print("type(d['A']): ", type(d['A']))
print(d.describe())

f = open("h1.csv", "w")
f.write(d.to_csv(index=True))
f.close()
print('ngon')
