from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import tensorflow.contrib.tensorrt as trt

import numpy as np
import h5py as h5
import time
from tensorflow.python.platform import gfile
from tensorflow.python.client import timeline
import argparse, sys, itertools,datetime
import json
tf.logging.set_verbosity(tf.logging.INFO)

import os

from utils import *


def printStats(graphName,timings,batch_size):
  if timings is None:
    return
  times=np.array(timings)
  speeds=batch_size / times
  avgTime=np.mean(timings)
  avgSpeed=batch_size/avgTime
  stdTime=np.std(timings)
  stdSpeed=np.std(speeds)
  print("images/s : %.1f +/- %.1f, s/batch: %.5f +/- %.5f"%(avgSpeed,stdSpeed,avgTime,stdTime))
  print("RES, %s, %s, %.2f, %.2f, %.5f, %.5f"%(graphName,batch_size,avgSpeed,stdSpeed,avgTime,stdTime))


#main function
if "__main__" in __name__:
  P=argparse.ArgumentParser(prog="test")
  P.add_argument('--FP32',action='store_true')
  P.add_argument('--FP16',action='store_true')
  P.add_argument('--INT8',action='store_true')
  P.add_argument('--input_prefix',type=str)
  P.add_argument('--input_path',type=str,default='./',help="path to read input files from for inference mode")
  P.add_argument('--num_loops',type=int,default=20)
  P.add_argument('--batch_size',type=int,default=128)
  P.add_argument('--with_timeline',action='store_true')
  P.add_argument('--gpu',type=int,default=0)  
  P.add_argument('--mode',type=str,default="time",help="Specify whether to run the graph or time it {time, run}")
  
  #parse args
  f,unparsed=P.parse_known_args()

  #select the GPU
  os.environ["CUDA_VISIBLE_DEVICES"]=str(f.gpu) #selects a specific device
  
  #inference
  valfp32=None
  valfp16=None
  valint8=None
  res=[None,None,None,None]
  timelineName=None
  print("Starting at",datetime.datetime.now())
  
  #hepcnn specific
  dummy_input = np.random.uniform(size=(f.batch_size, 3, 224, 224)).astype(np.float32)
  
  if f.FP32:
    #load graph
    graph_def = load_graph(f.input_prefix+'.FP32.pb')
    if f.mode == "time":
      if f.with_timeline: timelineName="FP32Timeline.json"
      timings,comp,valfp32,mdstats = timeGraph(graph_def, f.batch_size, f.num_loops, "Placeholder", ["Softmax"], dummy_input, timelineName)
      printStats("TRT-FP32",timings,f.batch_size)
      printStats("TRT-FP32RS",mdstats,f.batch_size)
    elif f.mode=='run':
      result = runGraph(graph_def, f.batch_size, 1, "Placeholder", ["Softmax"])
    elif f.mode == 'inference':
      outfilename = 'results_fp32.h5'
      predictions, labels, weights, psr = runGraph(graph_def, f.batch_size, 1, "Placeholder", ["Softmax"], dtype=np.float32, input_data=f.input_path)
    else:
      raise ValueError("Error, unknown mode {m}.".format(m=f.mode))
  if f.FP16:
    #load graph
    graph_def = load_graph(f.input_prefix+'.FP16.pb')
    if f.mode == "time":
      if f.with_timeline: timelineName="FP16Timeline.json"
      timings,comp,valfp32,mdstats = timeGraph(graph_def, f.batch_size, f.num_loops, "Placeholder", ["Softmax"], dummy_input, timelineName)
      printStats("TRT-16",timings,f.batch_size)
      printStats("TRT-FP16RS",mdstats,f.batch_size)
    elif f.mode == 'run':
      result = runGraph(graph_def, f.batch_size, 1, "Placeholder", ["Softmax"])
    elif f.mode == 'inference':
      outfilename = 'results_fp16.h5'
      predictions, labels, weights, psr = runGraph(graph_def, f.batch_size, 1, "Placeholder", ["Softmax"], dtype=np.float32, input_data=f.input_path)
    else:
      raise ValueError("Error, unknown mode {m}.".format(m=f.mode))

  if f.INT8:
    #load graph
    graph_def = load_graph(f.input_prefix+'.INT8.pb')
    if f.mode == "time":
      if f.with_timeline: timelineName="FP16Timeline.json"
      timings,comp,valint8,mdstats = timeGraph(graph_def, f.batch_size, f.num_loops, "Placeholder", ["Softmax"], dummy_input, timelineName)
      printStats("TRT-8",timings,f.batch_size)
      printStats("TRT-INT8RS",mdstats,f.batch_size)
    elif f.mode == 'run':
      result = runGraph(graph_def, f.batch_size, 1, "Placeholder", ["Softmax"])
    elif f.mode == 'inference':
      outfilename = 'results_int8.h5'
      predictions, labels, weights, psr = runGraph(graph_def, f.batch_size, 1, "Placeholder", ["Softmax"], dtype=np.float32, input_data=f.input_path)
    else:
      raise ValueError("Error, unknown mode {m}.".format(m=f.mode))
    
  if f.mode == 'inference':
    with h5.File(outfilename, 'w-') as f:
      f['prediction'] = predictions[...]
      f['label'] = labels[...]
      f['weight'] = weights[...]
      f['psr'] = psr[...]
