#!/bin/bash

#activate conda env
#source activate thorsten-tf-py27

#convert
python convert_tensorrt_tf_integrated.py --input_file="../model/NCHW/cosmoGAN_frozen.pb" --output_prefix="../model/NCHW/cosmoGAN_TRT" --batch_size=64 --FP32 --FP16 --INT8 --gpu 2

