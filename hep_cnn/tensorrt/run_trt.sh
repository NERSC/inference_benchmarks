#!/bin/bash

#FP32
#python run_tensorrt_tf_integrated.py --input_prefix="../model/trt/hep_cnn_bs_32" --batch_size 32 --gpu 2 --FP32 --mode time

#FP16
python run_tensorrt_tf_integrated.py --input_prefix="../model/trt/hep_cnn_bs_32" --batch_size 32 --gpu 2 --FP16 --mode time