#!/bin/bash

#input paths
inputdirreal=/data0/tkurth/data/cosmoGAN
inputdirfake=./results
outputdir=./plots

#create dirs
mkdir -p ${outputdir}

#run script
python ./evaluate.py --input_path_real=${inputdirreal} --input_path_fake=${inputdirfake} --output_path=${outputdir}
