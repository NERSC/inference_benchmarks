#!/bin/bash

#input paths
#inputdirreal=/data0/tkurth/data/cosmoGAN
inputdirreal=/data1/mustafa/cosmo/data
inputdirfake=./results
#inputdirfake=~/inference_benchmarks/cosmoGAN/generic
outputdir=./plots

#create dirs
mkdir -p ${outputdir}

#run script
python ./evaluate.py --input_path_real=${inputdirreal} --input_path_fake=${inputdirfake} --output_path=${outputdir}
