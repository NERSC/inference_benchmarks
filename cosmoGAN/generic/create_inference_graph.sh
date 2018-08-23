#!/bin/bash

#activate conda env
source activate thorsten-tf-py27

#some params
checkpoint_dir=/home/mustafa/cosmoGAN/networks/checkpoints/cosmo_myExp_batchSize64_flipLabel0.010_nd4_ng4_gfdim64_dfdim64_zdim64_2
input_graph=${checkpoint_dir}/dcgan.model-epoch-47.meta
input_weights=${checkpoint_dir}/dcgan.model-epoch-47

#creating inference graph
echo "Creating Inference Graph"
python inference_graph.py --input_meta_file=${input_graph} --input_checkpoint_file=${input_weights} --output_graph_file=../model/NCHW/cosmoGAN.pb --gpu 2

#freeeeeeze
echo "Freezing Graph"
python /home/tkurth/anaconda3/envs/thorsten-tf-py27/lib/python2.7/site-packages/tensorflow/python/tools/freeze_graph.py \
    --input_graph ../model/NCHW/cosmoGAN.pb \
    --input_binary \
    --input_checkpoint ${input_weights} \
    --output_node_names generator/Tanh \
    --output_graph ../model/NCHW/cosmoGAN_frozen.pb

#test the frozen graph again:
echo "Test Frozen Graph"
python test_inference_graph.py --input_graph_file=../model/NCHW/cosmoGAN_frozen.pb