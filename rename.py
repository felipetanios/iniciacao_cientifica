import os
import sys

for subdirs, dirs, files in os.walk('./'):
	for f in files:
		p = f
		for n in range(len(f)):
			k = n-1
			if p[k] == ' ':
				sec = (p[0:k], p[k+1:])
				p = "_".join(sec)
				print (p)
				print (f)
		os.rename(f, p)
		
