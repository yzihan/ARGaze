#!/usr/bin/env python
import numpy as np

import os 
import sys
import shutil


if len(sys.argv) != 2:
    print('Usage: {} <serial ID>'.format(sys.argv[0]))
    exit(1)

i_s = int(sys.argv[1])


data = np.load("serial{}/target.npy".format(i_s))

head_cut = 3000

if data.shape[0] > 32000:
    train = data[head_cut:30000]
else:
    train = data[head_cut:-2000]

size = train.shape[0]

def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def ln(src,dest):
    if not os.path.islink(dest):
        os.symlink(src, dest)

def mkserialdir(i):
    mkdir("serial{}".format(i))
    mkdir("serial{}/1".format(i))
    mkdir("serial{}/2".format(i))

def genserial(k, offset, length):
    mkserialdir(100 * i_s + k)
    dat = data[offset:offset + length]
    for i in range(dat.shape[0]):
        for j in [1,2]:
            ln('../../serial{}/{}/{}.png'.format(i_s, j, i + offset + head_cut), 'serial{}/{}/{}.png'.format(i_s * 100 + k, j, i))
    np.save("serial{}/target.npy".format(100 * i_s + k), dat)

half = size // 2
test_size = half // 10

train_size = half - test_size

genserial(6, 0, test_size)
genserial(7, test_size, train_size)
genserial(8, test_size + train_size, test_size)
genserial(9, test_size * 2 + train_size, train_size)
