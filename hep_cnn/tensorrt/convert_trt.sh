#!/bin/bash

#data directory
datadir=/data0/tkurth/data/hep_cnn/224

python convert_tensorrt_tf_integrated.py --input_path=${datadir}/validation --input_file="../model/original/hep_frozen_bs_32.pb" --output_prefix="../model/trt/hep_cnn_bs_32" --batch_size 32 --FP32 --FP16 --INT8 --gpu 2
