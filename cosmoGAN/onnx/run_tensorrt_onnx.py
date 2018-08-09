import tensorrt as trt

from tensorrt.parsers import onnxparser
apex = onnxparser.create_onnxconfig()

#create config object
apex.set_model_file_name("model/cosmoGAN.onnx")
apex.set_model_dtype(trt.infer.DataType.FLOAT)
apex.set_print_layer_info(True)

#create parser
trt_parser = onnxparser.create_onnxparser(apex)
data_type = apex.get_model_dtype()
onnx_filename = apex.get_model_file_name()

#parse
trt_parser.parse(onnx_filename, data_type)

#retrieve the network from the parser
trt_parser.convert_to_trtnetwork()
trt_network = trt_parser.get_trtnetwork()
