import random
with open('data.csv','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('shuffle.csv','w') as target:
    for _, line in data:
        target.write( line )