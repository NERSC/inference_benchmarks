from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import tensorflow.contrib.tensorrt as trt

import numpy as np
import time
from tensorflow.python.platform import gfile
from tensorflow.python.client import timeline
import argparse, sys, itertools,datetime
import json
tf.logging.set_verbosity(tf.logging.INFO)

import os

from utils import *

def getGraph(filename):
  with gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    return graph_def

def getFP32(input_file, output_prefix, output, batch_size=128, workspace_size=1<<20):
  trt_graph = trt.create_inference_graph(getGraph(input_file), output,
                                          max_batch_size=batch_size,
                                          max_workspace_size_bytes=workspace_size,
                                          precision_mode="FP32")  # Get optimized graph
  with gfile.FastGFile(output_prefix+'.FP32.pb', 'wb') as f:
    f.write(trt_graph.SerializeToString())
  return trt_graph

def getFP16(input_file, output_prefix, output, batch_size=128, workspace_size=1<<20):
  trt_graph = trt.create_inference_graph(getGraph(input_file), output,
                                         max_batch_size=batch_size,
                                         max_workspace_size_bytes=workspace_size,
                                         precision_mode="FP16")  # Get optimized graph
  with gfile.FastGFile(output_prefix+'.FP16.pb','wb') as f:
    f.write(trt_graph.SerializeToString())
  return trt_graph

def getINT8CalibGraph(input_file, output_prefix, output, batch_size=128, workspace_size=1<<20):
  trt_graph = trt.create_inference_graph(getGraph(input_file), output,
                                         max_batch_size=batch_size,
                                         max_workspace_size_bytes=workspace_size,
                                         precision_mode="INT8")  # calibration
  with gfile.FastGFile(output_prefix+'.INT8Calib.pb','wb') as f:
    f.write(trt_graph.SerializeToString())
  return trt_graph

def getINT8InferenceGraph(output_prefix, calibGraph):
  trt_graph=trt.calib_graph_to_infer_graph(calibGraph)
  with gfile.FastGFile(output_prefix+'.INT8.pb','wb') as f:
    f.write(trt_graph.SerializeToString())
  return trt_graph

#main
if "__main__" in __name__:
  
  P=argparse.ArgumentParser(prog="trt_convert")
  P.add_argument('--FP32',action='store_true')
  P.add_argument('--FP16',action='store_true')
  P.add_argument('--INT8',action='store_true')
  P.add_argument('--input_file',type=str)
  P.add_argument('--input_path_calibration',type=str,default='./',help="path to read input files from for calibration mode")
  P.add_argument('--output_prefix',type=str)
  P.add_argument('--batch_size',type=int, default=32)
  P.add_argument('--num_calibration_runs',type=int, default=100)
  P.add_argument('--workspace_size',type=int, default=1<<20,help="workspace size in MB")
  P.add_argument('--gpu', type=int, default=0)
  #P.add_argument('--update_graphdef',action='store_true')
  
  #parse args
  f,unparsed=P.parse_known_args()
  
  #select the GPU
  os.environ["CUDA_VISIBLE_DEVICES"]=str(f.gpu) #selects a specific device

  #create a session just in case
  sess = tf.Session()

  #print graph
  print_graph(f.input_file)
  
  #do the conversion
  if f.FP32:
    getFP32(input_file=f.input_file, output_prefix=f.output_prefix, output=["Softmax"], batch_size=f.batch_size, workspace_size=f.workspace_size)
  if f.FP16:
    getFP16(input_file=f.input_file, output_prefix=f.output_prefix, output=["Softmax"], batch_size=f.batch_size, workspace_size=f.workspace_size)
  if f.INT8:
    calibGraph = getINT8CalibGraph(input_file=f.input_file, output_prefix=f.output_prefix, output=["Softmax"], batch_size=f.batch_size, workspace_size=f.workspace_size)
    print('Calibrating Graph...')
    #run graph
    runGraph(calibGraph, f.batch_size, "Placeholder", ["Softmax"], dtype=np.float32, input_data=f.input_path_calibration)
    print('done...')
    #get int8 graph
    getINT8InferenceGraph(output_prefix=f.output_prefix, calibGraph=calibGraph)
    
  sys.exit(0)
