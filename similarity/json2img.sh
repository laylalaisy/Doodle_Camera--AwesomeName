#!/bin/bash

InputDir="./json/"
OutputDir="./image_ori/"

files=`ls $InputDir`

for file in $files
do
	label=$(echo $file | cut -d . -f1)
	mkdir $OutputDir$label
	image_folder=$OutputDir$label"/"
	python ./json2img.py $label $file $image_folder
done
