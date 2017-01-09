import os
import filecmp

for filename in os.listdir('./data/images'):
    if filecmp.cmp('./data/images/' + filename, './data/error.png'):
        os.remove('./data/images/' + filename)
