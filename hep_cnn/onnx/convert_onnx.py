import tensorflow as tf
from onnx_tf.frontend import tensorflow_graph_to_onnx_model

with tf.gfile.GFile("../model/original/hep_frozen_bs_64.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    #remove unnecessary stuff
    pruned_graph = tf.graph_util.remove_training_nodes(graph_def)
    print(type(pruned_graph))

    #write to model
    onnx_model = tensorflow_graph_to_onnx_model(graph_def=pruned_graph,
                                                output="Softmax",
                                                opset=0, ignore_unimplemented=True)

    with open("../model/onnx/hep_frozen_bs_64.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    for node in onnx_model.graph.node:
        print(node)
    #print(onnx_model.graph.node[-1])
