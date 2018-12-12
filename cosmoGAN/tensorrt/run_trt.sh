#!/bin/bash

#mode
mode=power
gpu=1

#number of samples
num_batches=400

#create output dir
mkdir -p results

#run fp32
if [ "${mode}" == "inference" ]; then
    echo "running fp32"
    python run_tensorrt_tf_integrated.py --input_prefix="../model/NCHW/cosmoGAN_TRT" --batch_size 64 --num_batches ${num_batches} --gpu ${gpu} --FP32 --mode inference
    mv result_fp32.h5 results/

    #run fp16
    echo "running fp16"
    python run_tensorrt_tf_integrated.py --input_prefix="../model/NCHW/cosmoGAN_TRT" --batch_size 64 --num_batches ${num_batches} --gpu ${gpu} --FP16 --mode inference
    mv result_fp16.h5 results/

    #run int8
    echo "running int8"
    python run_tensorrt_tf_integrated.py --input_prefix="../model/NCHW/cosmoGAN_TRT" --batch_size 64 --num_batches ${num_batches} --gpu ${gpu} --INT8 --mode inference
    mv result_int8.h5 results/
fi

if [ "${mode}" == "profile" ]; then
    echo "running fp32 profiling"
    python run_tensorrt_tf_integrated.py --input_prefix="../model/NCHW/cosmoGAN_TRT" --batch_size 64 --num_batches ${num_batches} --gpu ${gpu} --FP32 --mode time > results/performance_fp32.out 2>&1

    echo "running fp16 profiling"
    python run_tensorrt_tf_integrated.py --input_prefix="../model/NCHW/cosmoGAN_TRT" --batch_size 64 --num_batches ${num_batches} --gpu ${gpu} --FP16 --mode time > results/performance_fp16.out 2>&1

    echo "running int8 profiling"
    python run_tensorrt_tf_integrated.py --input_prefix="../model/NCHW/cosmoGAN_TRT" --batch_size 64 --num_batches ${num_batches} --gpu ${gpu} --INT8 --mode time > results/performance_int8.out 2>&1
fi

if [ "${mode}" == "power" ]; then
    echo "running fp32 power measurement"
    nvprof --system-profiling on --print-gpu-trace --csv python run_tensorrt_tf_integrated.py --input_prefix="../model/NCHW/cosmoGAN_TRT" --batch_size 64 --num_batches ${num_batches} --gpu ${gpu} --FP32 --mode time > results/nvprof_fp32.out 2>&1

    echo "running fp16 power measurement"
    nvprof --system-profiling on --print-gpu-trace --csv python run_tensorrt_tf_integrated.py --input_prefix="../model/NCHW/cosmoGAN_TRT" --batch_size 64 --num_batches ${num_batches} --gpu ${gpu} --FP16 --mode time > results/nvprof_fp16.out 2>&1

    echo "running int8 power measurement"
    nvprof --system-profiling on --print-gpu-trace --csv python run_tensorrt_tf_integrated.py --input_prefix="../model/NCHW/cosmoGAN_TRT" --batch_size 64 --num_batches ${num_batches} --gpu ${gpu} --INT8 --mode time > results/nvprof_int8.out 2>&1
fi
