#!/usr/bin/env python

import os
import sys
import pathlib
import shutil
import numpy as np
import cv2
import time

def mkdir(p: str):
	try:
		os.mkdir(p)
	except FileExistsError:
		pass

def cp(src: str, dst: str):
	try:
		shutil.copyfile(src, dst)
	except shutil.SameFileError:
		pass

def getinput(person: int, ser: int):
	d = 'dataset2_test{}/serial{}'.format(person, 3200 + ser)
	if not os.path.isdir(d):
		raise ValueError('Specified person or serie does not exist!')
	return d

def getinput2(person: int, ser: int):
#	d = 'dataset2_test{}/serial{}'.format(person, ser)
	d = '/tmp/memdisk/data_p{}s{}'.format(person, ser)
	if not os.path.isdir(d):
		raise ValueError('Specified person or serie does not exist!')
	return d

def getoutput(person: int, ser: int):
	d = '/tmp/memdisk/person{}_serie{}'.format(person, ser)
	pathlib.Path(d).mkdir(parents=True, exist_ok=True)

	mkdir(os.path.join(d, '0'))
	mkdir(os.path.join(d, '1'))
	mkdir(os.path.join(d, '2'))

	return d

if len(sys.argv) != 4:
	print('Usage: {} <personid> <in_serie_id> <out_serie_id>'.format(sys.argv[0]))
	exit(1)

person = int(sys.argv[1])
i_ser = int(sys.argv[2])
o_ser = int(sys.argv[3])


ip2 = getinput2(person, i_ser)

#cap = cv2.VideoCapture(os.path.join(ip2, '0/all.mp4'))
#if not cap.isOpened():
#	print('Video not opened', file=sys.stderr, flush=True)
#	exit(1)

ip = getinput(person, i_ser)
op = getoutput(person, o_ser)


data = np.load(os.path.join(ip, 'target.npy'))

start = 3000

end = data.shape[0] - 2000
if end > 30000:
	end = 30000
length = ((end - start - 3000) & (~511)) + 3000
end = start + length


#print('Frame size: {}x{} @ {}fps'.format(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)), flush=True)
#cap.set(cv2.CAP_PROP_POS_FRAMES, start)
#print('Start from frame#{}'.format(cap.get(cv2.CAP_PROP_POS_FRAMES)), flush=True)

#
#fail = 0
#pf = None
#

st = time.time()

for i in range(length):
	si = start + i
	di = i
	
#	s, frame = cap.read()
#	if not s:
#		fail += 1
#		print('Skipped', flush=True)
#		if fail > 5:
#			print('Too many failures while reading video, video corrupted.', file=sys.stderr, flush=True)
#			exit(1)
#		while pf is None and not s:
#			fail += 1
#			print('Skipped Again', flush=True)
#			if fail > 5:
#				print('Too many failures while reading video, video corrupted.', file=sys.stderr, flush=True)
#				exit(1)
#			s, frame = cap.read()
#		frame = pf
#	else:
#		pf = frame
#
#	cv2.imwrite(os.path.join(op, '0/{}.png'.format(di)), frame)

	cp(os.path.join(ip2, '{}.png'.format(si)), os.path.join(op, '0/{}.png'.format(di)))
	cp(os.path.join(ip, '1/{}.png'.format(si)), os.path.join(op, '1/{}.png'.format(di)))
	cp(os.path.join(ip, '2/{}.png'.format(si)), os.path.join(op, '2/{}.png'.format(di)))

	ct = time.time()
	if ct - st > 10:
		print('{}/{}'.format(i + 1, length), end='  \r', flush=True)

print('{}/{}'.format(i + 1, length), flush=True)

np.save(os.path.join(op, 'target.npy'), data[start: end])

print('Done!')

#cap.release()
