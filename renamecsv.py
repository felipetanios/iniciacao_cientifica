import csv
with open('base.csv', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
        	f = row[0]
        	for n in range(len(f)):
			k = n-1
			if f[k] == ' ':
				sec = (f[0:k], f[k+1:])
				n = "_".join(sec)
				f = n