#!/bin/bash

InputDir="./ndjson/"
OutputDir="./json/"

files=`ls $InputDir`
for file in $files
do
	filename=$(echo $file | cut -d . -f1)
	ndjson_filename=$InputDir$filename".ndjson"
	json_filename=$OutputDir$filename".json"
	node ./simplified-parser.js $ndjson_filename $json_filename
done
