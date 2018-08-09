#!/bin/bash

python run_tensorrt_tf_integrated.py --input_prefix="../model/cosmoGAN_TRT" --batch_size 64 --gpu 2 --FP32 --mode time

python run_tensorrt_tf_integrated.py --input_prefix="../model/cosmoGAN_TRT" --batch_size 64 --gpu 2 --FP16 --mode time

