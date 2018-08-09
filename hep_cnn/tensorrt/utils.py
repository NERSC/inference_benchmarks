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
import data

def display_nodes(nodes):
  for i, node in enumerate(nodes):
    print('%d %s %s' % (i, node.name, node.op))
    [print(u'\'--- {} - {}'.format(i, n)) for i, n in enumerate(node.input)]

    
def load_graph(filename):
  with gfile.FastGFile(filename,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


def print_graph(filename):
  gdef = load_graph(filename)

  #display
  display_nodes(gdef.node)


#time graph function
def timeGraph(gdef, batch_size, num_loops, input_name, outputs, dummy_input, timelineName=None):
  
  tf.logging.info("Starting execution")
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
  tf.reset_default_graph()
  g = tf.Graph()
  outlist=[]
  with g.as_default():
    dataset=tf.data.Dataset.from_tensor_slices(dummy_input)
    dataset=dataset.repeat()
    dataset=dataset.batch(batch_size)
    iterator=dataset.make_one_shot_iterator()
    next_element=iterator.get_next()
    out = tf.import_graph_def(
      graph_def=gdef,
      input_map={input_name:next_element},
      return_elements=outputs
    )
    out = out[0].outputs[0]
    outlist.append(out)
    
  timings=[]
  
  with tf.Session(graph=g,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    tf.logging.info("Starting Warmup cycle")
    def mergeTraceStr(mdarr):
      tl=timeline.Timeline(mdarr[0][0].step_stats)
      ctf=tl.generate_chrome_trace_format()
      Gtf=json.loads(ctf)
      deltat=mdarr[0][1][1]
      for md in mdarr[1:]:
        tl=timeline.Timeline(md[0].step_stats)
        ctf=tl.generate_chrome_trace_format()
        tmp=json.loads(ctf)
        deltat=0
        Gtf["traceEvents"].extend(tmp["traceEvents"])
        deltat=md[1][1]
        
      return json.dumps(Gtf,indent=2)
    
    rmArr=[[tf.RunMetadata(),0] for x in range(20)]
    if timelineName:
      if gfile.Exists(timelineName):
        gfile.Remove(timelineName)
      ttot=int(0)
      tend=time.time()
      for i in range(20):
        tstart=time.time()
        valt = sess.run(outlist, options=run_options, run_metadata=rmArr[i][0])
        tend=time.time()
        rmArr[i][1]=(int(tstart*1.e6),int(tend*1.e6))
      with gfile.FastGFile(timelineName,"a") as tlf:
        tlf.write(mergeTraceStr(rmArr))
    else:
      for i in range(20):
        valt = sess.run(outlist)
    tf.logging.info("Warmup done. Starting real timing")
    num_iters=50
    
    for i in range(num_loops):
      tstart=time.time()
      for k in range(num_iters):
        val = sess.run(outlist)
      timings.append((time.time()-tstart)/float(num_iters))
      print("iter ",i," ",timings[-1])
    comp=sess.run(tf.reduce_all(tf.equal(val[0],valt[0])))
    print("Comparison=",comp)
    sess.close()
    tf.logging.info("Timing loop done!")
    return timings,comp,val[0],None


#produce output
def runGraph(gdef, batch_size, input_name, outputs, dtype=np.float32, input_data=None):
  
  #set up graph
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
  tf.reset_default_graph()
  g = tf.Graph()
  outlist=[]
  
  with g.as_default():
    #input
    if not input_data:
      input_data = np.random.uniform(size=(batch_size*100, 3, 224, 224)).astype(dtype)
      dataset=tf.data.Dataset.from_tensor_slices(input_data)
      dataset=dataset.repeat(1)
      dataset=dataset.batch(batch_size)
      iterator=dataset.make_one_shot_iterator()
      next_image=iterator.get_next()
      
    elif isinstance(input_data, str): 
      #scan input path
      filelist=sorted([os.path.join(input_data, x) for x in os.listdir(input_data) if x.endswith('.h5')])
      
      #instantiate reader:
      h5ir = data.DataSet(filelist,dtype=dtype)
      
      if dtype == np.float32:
        tftype = tf.float32
      elif dtype == np.float16:
        tftype = tf.float16
      else:
        raise ValueError("Error, type {dt} not supported.".format(dt=dtype))
      
      #create dataset
      dataset = tf.data.Dataset.from_generator(h5ir.next, 
                                              output_types = (tftype, tf.int32, tf.float32, tf.float32, tf.int32), 
                                              output_shapes = ((3, 224, 224), (1), (1), (1), (1)))
      dataset = dataset.prefetch(batch_size)
      dataset = dataset.batch(batch_size, drop_remainder=True)
      iterator = dataset.make_one_shot_iterator()
      next_element = iterator.get_next()
      next_image = next_element[0]
    
    out = tf.import_graph_def(
      graph_def=gdef,
      input_map={input_name:next_image},
      return_elements=outputs
    )
    out = out[0].outputs[0]
    outlist.append(out)
    outlist.append(next_element[1])
    outlist.append(next_element[3])
    outlist.append(next_element[4])
    
  with tf.Session(graph=g,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    
    predictions=[]
    labels=[]
    weights=[]
    psr=[]
    
    #loop over dataset
    while True:
      try:
        vals = sess.run(outlist)
        predictions.append(vals[0][:,1])
        labels.append(vals[1][:,0])
        weights.append(vals[2][:,0])
        psr.append(vals[3][:,0])
      except:
        print("I am done.")
        predictions = np.concatenate(predictions, axis=0)
        labels = np.concatenate(labels, axis=0)
        weights = np.concatenate(weights, axis=0)
        psr = np.concatenate(psr, axis=0)
        break
        
  return predictions, labels, weights, psr
