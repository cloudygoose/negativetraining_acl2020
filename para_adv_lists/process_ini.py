from __future__ import print_function

import sys, os

import argparse

parser = argparse.ArgumentParser(description='data-preprocessing')

parser.add_argument('--file', type=str, default=None, required=True, help='file to be processed')
parser.add_argument('--vocab_file', type=str, default=None, required=True)
parser.add_argument('--out_train_file', type=str, default=None, required=True)
parser.add_argument('--out_test_file', type=str, default=None, required=True)
parser.add_argument('--out_pair_file', type=str, default=None, required=True)

args = parser.parse_args()

vocab = {}
print('loading vocab %s...' % args.vocab_file)
for line in open(args.vocab_file).readlines():
    vocab[line.strip()] = True

train_lis, test_lis, pair_lis = [], [], []
dd = {}
print('processing file %s...' % args.file)
l_co = 0
for line in open(args.file).readlines():
    line = line.strip()
    l_co = l_co + 1
    if len(line) < 2: continue
    for w in line.split():
        if len(w) < 1: continue
        if not w in vocab:
            print('line_co: %d not_in_vocab error! w: %s str: %s' % (l_co, w, line))
    terms = line.split('\t')    
    terms = [s for s in terms if len(s) > 1]
    assert(len(terms) == 2)
    train_lis.append(terms[0]); test_lis.append(terms[1]);
    for ss in [terms[0], terms[1]]:
        if ss in dd:
            print('dup!: %s' % ss)
        dd[ss] = True;
    pair_lis.append(terms[0] + ' <eou> ' + terms[1])

print('vocab check okay, writing to %s and %s and %s' % (args.out_train_file, args.out_test_file, args.out_pair_file))
open(args.out_train_file, 'w').write('\n'.join(train_lis) + '\n')
open(args.out_test_file, 'w').write('\n'.join(test_lis) + '\n')
open(args.out_pair_file, 'w').write('\n'.join(pair_lis) + '\n')
sys.exit(0)
