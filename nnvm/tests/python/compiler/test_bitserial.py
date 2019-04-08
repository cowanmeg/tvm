import numpy as np
import tvm
from tvm.contrib import graph_runtime
import topi.testing
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list
from nnvm.compiler import graph_util
from tvm.contrib import graph_runtime as runtime
 
input_dtype = 'uint8'
output_dtype = 'int16'

def _bitserial_conv2d(channels, kernel_size, padding, activation_bits, weight_bits):
    data = sym.Variable(name="data")
    data = sym.cast(data=data, dtype=input_dtype)
    bitserial_conv2d = sym.bitserial_conv2d(data=data, kernel_size=kernel_size, channels=channels, 
                        padding=padding,
                        layout="NHWC", kernel_layout="HWIO", 
                        activation_bits=activation_bits, weight_bits=weight_bits,
                        out_dtype=output_dtype, pack_dtype='uint8', name='bitserial_conv2d')
    return bitserial_conv2d


def build_and_run(sym, data, out_shape, graph, lib, params, output_type):
    ctx = tvm.cpu(0)
    module = runtime.create(graph, lib, ctx)
    module.set_input(**params)
    module.set_input("data", data)
    module.run()
    out =  module.get_output(0, tvm.nd.empty(out_shape, output_type))
    return out.asnumpy()

# Test that bitserial methods give same result as standard convolutions given identical input
def test_bitserial_conv():
    data_shape = (1, 56, 56, 64)
    out_channel = 64
    filter_shape = (3, 3, 64, out_channel)
    output_shape = (1, 56, 56, out_channel)
    kernel_size = (3, 3)
    padding = (1, 1, 1, 1)
    activation_bits = 2
    weight_bits = 1

    bitserial_conv2d_sym = _bitserial_conv2d(out_channel, kernel_size, padding,
        activation_bits, weight_bits)


    conv_weight = np.random.randint(0, 2 ** weight_bits, size=filter_shape).astype("uint8")

    bitserial_params = {
        "bitserial_conv2d_weight" : tvm.nd.array(conv_weight, ctx=tvm.cpu(0)),
    }
    data = np.random.randint(0, 2 ** activation_bits, size=data_shape).astype("uint8")


    graph, lib, params = nnvm.compiler.build(bitserial_conv2d_sym, "llvm", 
        dtype={"data": input_dtype, "bitserial_conv2d_weight": input_dtype}, 
        shape={"data":data.shape}, 
        params=bitserial_params)

    bitserial_out = quantized__output = build_and_run(bitserial_conv2d_sym, data, output_shape, 
        graph, lib, params, output_dtype)


if __name__ == "__main__":
    test_bitserial_conv()
