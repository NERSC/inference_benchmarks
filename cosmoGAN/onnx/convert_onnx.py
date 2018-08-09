import tensorflow as tf
from onnx_tf.frontend import tensorflow_graph_to_onnx_model

with tf.gfile.GFile("model/cosmoGAN_frozen.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    onnx_model = tensorflow_graph_to_onnx_model(graph_def=graph_def,
                                                output="generator/Tanh",
                                                opset=0, ignore_unimplemented=True)

    file = open("model/cosmoGAN.onnx", "wb")
    file.write(onnx_model.SerializeToString())
    file.close()
