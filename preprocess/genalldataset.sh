#!/bin/bash

for id in *
do
	if [[ -d $id ]]
	then
		for seq in `seq 1 1 8`
		do
			if [[ -f $id/cut$seq.mp4 ]]
			then
				./gendataset.py $id $seq $1
				# || exit 1
			fi
		done
	fi
done
