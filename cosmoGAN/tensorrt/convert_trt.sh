#!/bin/bash

python convert_tensorrt_tf_integrated.py --input_file="../model/cosmoGAN_frozen.pb" --output_prefix="../model/cosmoGAN_TRT" --batch_size=64 --FP32 --FP16 --INT8 --gpu 2

