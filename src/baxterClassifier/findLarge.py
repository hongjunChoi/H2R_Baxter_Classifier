import glob

for filename in glob.iglob('tmp/*/*'):
    print(filename)