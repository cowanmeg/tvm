import numpy as np
import tvm
from tvm.contrib import graph_runtime
import topi.testing
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list
from nnvm.compiler import graph_util
from tvm.contrib import graph_runtime as runtime
 
def _bitserial_conv2d_nchw(channels, kernel_size, padding, activation_bits, weight_bits):
    data = sym.Variable(name="data")
   # quantized_data = data * ((1 << activation_bits) - 1) + 0.5
    data = sym.cast(data=data, dtype='int32')
    bitserial_conv2d = sym.bitserial_conv2d(data=data, kernel_size=kernel_size, channels=channels, 
                        padding=padding,
                        layout="NCHW", kernel_layout="OIHW", use_bias=False, 
                        activation_bits=activation_bits, weight_bits=weight_bits,
                        out_dtype='int32', name='bitserial_conv2d')
    bitserial_conv2d = sym.cast(bitserial_conv2d, dtype='float32', name='cast_bitserial_conv2d')
    #bitserial_conv2d_scaled = bitserial_conv2d * (1.0 / (((1<<activation_bits)-1)*((1<<weight_bits)-1)))
    return bitserial_conv2d

def _conv2d_nchw(channels, kernel_size, padding):
    data = sym.Variable(name="data")
    conv2d = sym.conv2d(data=data, kernel_size=kernel_size, channels=channels, padding=padding,
                      layout="NCHW", kernel_layout="OIHW", use_bias=False, name='conv2d')
    return conv2d


def build_and_run(sym, data, out_shape, graph, lib, params, output_type):
    ctx = tvm.cpu(0)
    module = runtime.create(graph, lib, ctx)
    module.set_input(**params)
    module.set_input("data", data)
    module.run()
    out =  module.get_output(0, tvm.nd.empty(out_shape, output_type))
    return out.asnumpy()

# Test that bitserial methods give same result as standard convolutions given identical input
def test_bitserial_conv_nchw():
    # data_shape = (1, 96, 27, 27)
    # out_channel = 192
    # filter_shape = (out_channel, 96, 5, 5)
    # output_shape = (1, out_channel, 27, 27)
    # kernel_size = (5, 5)
    # padding = (2, 2)

    data_shape = (1, 64, 56, 56)
    out_channel = 64
    filter_shape = (out_channel, 64, 3, 3)
    output_shape = (1, out_channel, 56, 56)
    kernel_size = (3, 3)
    padding = (1, 1)
    activation_bits = 2
    weight_bits = 1

    bitserial_conv2d_sym = _bitserial_conv2d_nchw(out_channel, kernel_size, padding,
        activation_bits, weight_bits)
    conv2d_sym = _conv2d_nchw(out_channel, kernel_size, padding)

    conv_weight = np.random.randint(0, 2 ** weight_bits, size=filter_shape).astype("int32")
    conv_bias = np.random.uniform(size=(out_channel)).astype("float32")

    params = {
        "conv2d_weight" : tvm.nd.array(conv_weight.astype('float32'), ctx=tvm.cpu(0)),
        "conv2d_bias" : tvm.nd.array(conv_bias.astype('float32'), ctx=tvm.cpu(0))
    }
    bitserial_params = {
        "bitserial_conv2d_weight" : tvm.nd.array(conv_weight, ctx=tvm.cpu(0)),
        "bitserial_conv2d_bias" : tvm.nd.array(conv_bias, ctx=tvm.cpu(0))
    }
    data = np.random.randint(0, 2 ** activation_bits, size=data_shape).astype("float32")

    graph, lib, params = nnvm.compiler.build(conv2d_sym, "llvm", 
        dtype={"data":"float32", "conv2d_weight":"float32", "conv2d_bias":"float32"}, 
        shape={"data":data.shape}, 
        params=params)
    out = _output = build_and_run(conv2d_sym, data, output_shape, graph, lib, params, 'float32')

    graph, lib, params = nnvm.compiler.build(bitserial_conv2d_sym, "llvm", 
        dtype={"data":"float32", "bitserial_conv2d_weight":"int32", "qconv2d0_bias":"float32"}, 
        shape={"data":data.shape}, 
        params=bitserial_params)

    bitserial_out = quantized__output = build_and_run(bitserial_conv2d_sym, data, output_shape, 
        graph, lib, params, "float32")

    np.testing.assert_allclose(out, bitserial_out, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_bitserial_conv_nchw()
