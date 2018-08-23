#!/bin/bash

#source conda
source activate thorsten-tf-py27 

#number of samples
num_batches=400

#create output dir
mkdir -p results

#run fp32
python run_tensorrt_tf_integrated.py --input_prefix="../model/NCHW/cosmoGAN_TRT" --batch_size 64 --num_batches ${num_batches} --gpu 2 --FP32 --mode inference
mv result_fp32.h5 results/

#run fp16
python run_tensorrt_tf_integrated.py --input_prefix="../model/NCHW/cosmoGAN_TRT" --batch_size 64 --num_batches ${num_batches} --gpu 2 --FP16 --mode inference
mv result_fp16.h5 results/
