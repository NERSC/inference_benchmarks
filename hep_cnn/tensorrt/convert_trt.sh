#!/bin/bash

python convert_tensorrt_tf_integrated.py --input_file="../model/original/hep_frozen_bs_32.pb" --output_prefix="../model/trt/hep_cnn_bs_32" --batch_size 32 --FP32 --FP16 --INT8 --gpu 2
