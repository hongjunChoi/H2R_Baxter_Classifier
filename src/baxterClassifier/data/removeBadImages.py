import os
import filecmp

count = 0
for filename in os.listdir('./images'):
	print(filename)
	if filecmp.cmp('./images/'+filename, './error.png'):
		count+=1
		os.remove('./images/'+filename)

print count