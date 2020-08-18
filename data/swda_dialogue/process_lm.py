import sys
import random

print >> sys.stderr, 'processing', sys.argv[1]

lines = open(sys.argv[1]).readlines()

for line in lines:
    line = line.strip().lower().split('<eou>')
    for ww in line:
        if (len(ww) < 1): continue
        while (len(ww) > 0 and ww.startswith(' ')): ww = ww[1:]
        while (len(ww) > 0 and ww.endswith(' ')): ww = ww[:-1]
        if len(ww) < 1: continue
        print ww

