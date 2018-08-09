#!/bin/bash

#data directory
datadir=/data0/tkurth/data/hep_cnn/224

#FP32
python run_tensorrt_tf_integrated.py --input_prefix="../model/trt/hep_cnn_bs_32" --input_path=${datadir} --batch_size 32 --gpu 2 --FP32 --mode inference
mv results_fp32.h5 results/

#FP16
python run_tensorrt_tf_integrated.py --input_prefix="../model/trt/hep_cnn_bs_32" --input_path=${datadir} --batch_size 32 --gpu 2 --FP16 --mode inference
mv results_fp16.h5 results/
