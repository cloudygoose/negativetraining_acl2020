import sys
import operator

dd = {}

for line in open(sys.argv[1], 'r'):
    line = line.strip().lower().split()
    for w in line:
        if w in dd:
            dd[w] = dd[w] + 1
        else:
            dd[w] = 1

s = sorted(dd.items(), key=operator.itemgetter(1), reverse = True)

for i in range(len(s)):
    print s[i][0], s[i][1]

