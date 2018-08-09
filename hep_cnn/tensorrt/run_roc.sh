#!/bin/bash

#input paths
inputdirhdf5=./results
outputdir=./plots

#create dirs
mkdir -p ${outputdir}

#run script
python ./evaluate_roc.py --input_path=${inputdirhdf5} --output_path=${outputdir}
