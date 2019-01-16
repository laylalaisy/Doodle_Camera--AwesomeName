#!/bin/bash

InputDir="./json/"
OutputDir="./image_orig/"

files=`ls $InputDir`

for file in $files
do
	label=$(echo $file | cut -d . -f1)
	mkdir $OutputDir$label
	json_file=$InputDir$file
	image_folder=$OutputDir$label"/"
	python ./json2img.py $label $json_file $image_folder
done
