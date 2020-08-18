from __future__ import print_function

import sys, os

import argparse

parser = argparse.ArgumentParser(description='train file augment')

parser.add_argument('--train_file', type=str, default=None, required=True, help='file to be processed')
parser.add_argument('--para_file', type=str, default=None, required=True)
parser.add_argument('--out_train_file', type=str, default=None, required=True)
parser.add_argument('--aug_num', type=int, default=None, required=True)

args = parser.parse_args()

out_lis = []
para_lines = open(args.para_file, 'r').readlines()
id_now = 0
for line in open(args.train_file, 'r').readlines():
    out_lis.append(line.strip())
    ps = para_lines[id_now].strip().split('\t')
    for i in range(args.aug_num - 1):
        out_lis.append(ps[i])
    id_now = id_now + 1

open(args.out_train_file, 'w').write('\n'.join(out_lis) + '\n')
