from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.ops import data_flow_ops
import tensorflow.contrib.tensorrt as trt
import os, sys
import argparse

sys.path.append("/home/tkurth/inference_benchmarks/cosmoGAN/cosmoGAN_src/networks")
from models import dcgan, utils
import numpy as np
import h5py as h5

#main
def main(params):

  graph_def = tf.GraphDef()
  with tf.gfile.FastGFile(params.input_graph_file,'rb') as f:
    graph_def.ParseFromString(f.read())
    
  with tf.Session() as sess:
    z, output = tf.import_graph_def(graph_def, return_elements=['z:0','generator/Tanh:0'], name='')
    
    #obtain results      
    results = []
    for i in range(0,params.num_tests):
      val = sess.run(output, feed_dict={z: np.random.normal(size=(64,64))})
      results.append(np.squeeze(val))
        
    #done
    results = np.concatenate(results, axis=0)
    with h5.File(params.output_test_file,"w") as f:
      f["data"] = results
    

if "__main__" in __name__:
  
  P=argparse.ArgumentParser(prog="inference_graph")
  P.add_argument('--input_graph_file',type=str,required=True,help="directory of graph file to reload")
  P.add_argument('--num_tests',  type=int, default=200)
  P.add_argument('--output_test_file',type=str,default="test.h5",help="output filename of tested inference graph")
  P.add_argument('--gpu', type=int, default=0)
  #P.add_argument('--update_graphdef',action='store_true')
  
  #parse args
  f,unparsed=P.parse_known_args()
  
  #select the GPU
  os.environ["CUDA_VISIBLE_DEVICES"]=str(f.gpu) #selects a specific device
  
  #run main and hand over the parameters
  main(f)
