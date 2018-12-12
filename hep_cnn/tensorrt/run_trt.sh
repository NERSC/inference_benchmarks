#!/bin/bash

#data directory
datadir=/mnt/hep_cnn/224
batchsize=32

#FP32
python run_tensorrt_tf_integrated.py --input_prefix="../model/trt/hep_cnn_bs_${batchsize}" --input_path=${datadir}/test --batch_size ${batchsize} --gpu 2 --FP32 --mode inference
mv results_fp32.h5 results/

#FP16
python run_tensorrt_tf_integrated.py --input_prefix="../model/trt/hep_cnn_bs_${batchsize}" --input_path=${datadir}/test --batch_size ${batchsize} --gpu 2 --FP16 --mode inference
mv results_fp16.h5 results/

#INT8
python run_tensorrt_tf_integrated.py --input_prefix="../model/trt/hep_cnn_bs_${batchsize}" --input_path=${datadir}/test --batch_size ${batchsize} --gpu 2 --INT8 --mode inference
mv results_int8.h5 results/
