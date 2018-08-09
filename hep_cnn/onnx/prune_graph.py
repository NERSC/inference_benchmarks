from __future__ import print_function
from tensorflow.core.framework import graph_pb2
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('%d %s %s' % (i, node.name, node.op))
        [print(u'\'--- {} - {}'.format(i, n)) for i, n in enumerate(node.input)]

        
def print_graph(filename):
    with tf.gfile.Open(filename, 'rb') as f:
        data = f.read()
        graph.ParseFromString(data)

    #display
    display_nodes(graph.node)    
        
        
# read frozen graph and display nodes
graph = tf.get_default_graph().as_graph_def(add_shapes=True)
with tf.gfile.Open('model/original/hep_frozen_bs_32.pb', 'rb') as f:
    data = f.read()
    graph.ParseFromString(data)
    
    #full graph
    #display_nodes(graph.node)
    
    #prune graph
    
    #first step
    graph.node[25].input[0] = 'Relu'
    nodes = graph.node[:12] + graph.node[23:]
    graph = graph_pb2.GraphDef()
    graph.node.extend(nodes)
    
    #next step
    graph.node[33].input[0] = 'Relu_1'
    nodes = graph.node[:20] + graph.node[31:]
    graph = graph_pb2.GraphDef()
    graph.node.extend(nodes)
    
    #next step
    graph.node[41].input[0] = 'Relu_2'
    nodes = graph.node[:28] + graph.node[39:]
    graph = graph_pb2.GraphDef()
    graph.node.extend(nodes)
    
    #next step
    graph.node[49].input[0] = 'Relu_3'
    nodes = graph.node[:36] + graph.node[47:]
    graph = graph_pb2.GraphDef()
    graph.node.extend(nodes)
    
    #delete some additional rap
    nodes = graph.node
    del nodes[1]
    
    #create close to final graph
    graph = graph_pb2.GraphDef()
    graph.node.extend(nodes)
    
    #now get rid of the input reshape, we will pass the right shape!
    graph.node[5].input[0] = 'Placeholder'
    nodes = graph.node[:1] + graph.node[3:]
    graph = graph_pb2.GraphDef()
    graph.node.extend(nodes)
    
    #create final graph
    graph = graph_pb2.GraphDef()
    graph.node.extend(nodes)
    
    #graph with dropout removed
    display_nodes(graph.node)
    
    with tf.gfile.GFile('model/pruned/hep_frozen_nodrop_bs_32.pb', 'w') as f:
        f.write(graph.SerializeToString())
