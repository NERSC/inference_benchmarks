#!/bin/bash

#data directory
datadir=/mnt/hep_cnn/224
num_calibration_files=1
num_calibration_runs=10
batchsize=32

#create directory for running calibration against, allows to select subset of input
rm -rf validation_tmp
mkdir -p validation_tmp
for vfile in $(ls ${datadir}/validation | head -n ${num_calibration_files}); do
    ln -s ${datadir}/validation/${vfile} validation_tmp/${vfile}
done
python convert_tensorrt_tf_integrated.py --input_path_calibration=validation_tmp --input_file="../model/original/hep_frozen_bs_${batchsize}.pb" --output_prefix="../model/trt/hep_cnn_bs_${batchsize}" --batch_size=${batchsize} --num_calibration_runs=${num_calibration_runs} --FP32 --FP16 --INT8 --gpu 2
