"""NNVM network definition of 2-bit activation 1-bit weight VGGNet.
"""

import numpy as np
import os
import tvm
import math
import pickle
import topi.testing
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list
from nnvm.compiler import graph_util
from  tvm import autotvm
from tvm.contrib.debugger import debug_runtime as graph_runtime

from PIL import Image
from tvm.contrib import util, rpc
from topi.nn.bitserial_conv2d import bitpack
from topi.util import get_const_tuple

from nnvm.testing.utils import create_workload


# Network settings
abits = 1
wbits = 1
layout = 'NHWC'
kernel_layout = 'HWIO'
kernel_bit_axis = 2
kernel_pack_axis = 2
kernel_pack_dtype = 'uint8'
output_dtype = 'float32'
bitpack_weights = True
if bitpack_weights:
    bit_axis = kernel_bit_axis
else:
    bit_axis = -1

def get_network():
    params = {}
    dtypes = {}

    d = '/sampa/home/cowanmeg/tvm-current/end2end/models/'
    old_dtypes = pickle.load(open(os.path.join(d, "vggnet_dtypes.p"), 'rb'))

    loaded_params = bytearray(open(os.path.join(d, "rasp_vggnet.params"), "rb").read())
    old_params = nnvm.compiler.load_param_dict(loaded_params)

    for k, v in old_params.items():
        if 'scalu' in k:
            params[k] = v
        print (k, v.shape)


    for k, v in old_dtypes.items():
        if 'scalu' in k:
            dtypes[k] = v
        print (k, v)

    def binary_conv(network, kernel_size, in_channels, channels, padding, name, max_pool=False):
        network = sym.bitserial_conv2d(data=network, kernel_size=(kernel_size, kernel_size), channels=channels, padding=padding, strides=(1, 1), layout=layout, 
            kernel_layout=kernel_layout, use_pool=False, activation_bits=abits, weight_bits=wbits, pack_dtype='uint8', out_dtype='int16', bit_axis=bit_axis, name=name)
        if max_pool:
            network = sym.max_pool2d(network, layout=layout, pool_size=(2, 2), padding=(0, 0), strides=(2, 2))
        r = sym.Variable(name+'round', shape=(channels, ))
        network = network + r
        sym.clip_channelwise(network, axis=3, name=name+'clip')
        network = sym.right_shift_channelwise(network, axis=3, name=name+'shift')
        params[name+'_weight'] = np.random.random_integers(1, 10, (kernel_size, kernel_size, wbits, in_channels//8, channels)).astype('uint8')
        dtypes[name+'_weight'] = 'uint8'
        params[name+'round'] = np.random.random_integers(1, 10, (channels, )).astype('int16')
        dtypes[name+'round'] = 'int16'
        params[name+'clip_a_min'] = np.random.random_integers(1, 10, (channels, )).astype('int16')
        dtypes[name+'clip_a_min'] = 'int16'
        params[name+'clip_a_max'] = np.random.random_integers(1, 10, (channels, )).astype('int16')
        dtypes[name+'clip_a_max'] = 'int16'
        return network

    def binary_dense(network, units, inputs, name):
        network = sym.flatten(data=network)
        network = sym.bitserial_dense(data=network, units=units, activation_bits=abits, weight_bits=wbits, pack_dtype='uint8', out_dtype='int16', name=name)
        r = sym.Variable(name+'round', shape=(units, ))
        network = network + r
        network = sym.clip_channelwise(network, axis=0, name=name+'clip')
        network = sym.right_shift_channelwise(network, axis=0, name=name+'shift')
        params[name+'_weight'] = np.random.random_integers(1, 10, (units, wbits, inputs//8)).astype('uint8')
        dtypes[name+'_weight'] = 'uint8'
        params[name+'round'] = np.random.random_integers(1, 10, (units, )).astype('int16')
        dtypes[name+'round'] = 'int16'
        params[name+'clip_a_min'] = np.random.random_integers(1, 10, (1, )).astype('int16')
        dtypes[name+'clip_a_min'] = 'int16'
        params[name+'clip_a_max'] = np.random.random_integers(1, 10, (1, )).astype('int16')
        dtypes[name+'clip_a_max'] = 'int16'
        return network

    network = sym.Variable(name='data')
    network = sym.conv2d(data=network, channels=64, kernel_size=(11, 11), strides=(4, 4), padding=[5, 5, 5, 5], layout=layout, kernel_layout=kernel_layout, use_bias=False, name='conv2d')
    network = sym.max_pool2d(network, name='max_pooling2d', layout=layout, pool_size=(2, 2), padding=(0, 0), strides=(2, 2))
    scale = sym.Variable('batch_normalization_scale', shape=(64,))
    shift = sym.Variable('batch_normalization_shift', shape=(64,))
    network = sym.broadcast_add(sym.broadcast_mul(network, scale), shift)
    network = network * 1
    network = sym.clip(data=network, a_min=0.0, a_max=1.0)
    data = network * ((1 << abits) - 1) + 0.5
    network = sym.cast(data=data, dtype='int16')
    params['conv2d'] = np.random.uniform((11, 11, 3, 64)).astype('float32')
    dtypes['conv2d'] = 'float32'
    params['batch_normalization_scale'] = np.random.uniform((64, )).astype('float32')
    params['batch_normalization_shift'] = np.random.uniform((64, )).astype('float32')
    dtypes['batch_normalization_scale'] = 'float32'
    dtypes['batch_normalization_shift'] = 'float32'

    network = binary_conv(network, 5, 64, 192, [2, 2, 2, 2], 'binary_conv2d', max_pool=True)
    network = binary_conv(network, 3, 192, 384, [1, 1, 1, 1], 'binary_conv2d1', max_pool=False)
    network = binary_conv(network, 3, 384, 384, [1, 1, 1, 1], 'binary_conv2d2', max_pool=False)
    network = binary_conv(network, 3, 384, 256, [1, 1, 1, 1], 'binary_conv2d3', max_pool=True)


    network = binary_dense(network, 4096, 7*7*256, 'binary_dense')
    network = binary_dense(network, 4096, 4096, 'binary_dense1')
    network = binary_dense(network, 1000, 4096, 'binary_dense2')

    scale = sym.Variable('scalu', shape=(1, 1000))
    network = sym.cast(data=network, dtype='float32')
    network = network * scale

    input_shape = (1, 224, 224, 3)
    output_shape = (1, 1000)

    return network, dtypes, params, input_shape, output_shape

def load_test_image():
    data_np = np.random.uniform(size=((1, 224, 224, 3))).astype('float32')
    data_tvm = tvm.nd.array(data_np)
    return data_np, data_tvm

def run(data, ctx):
    module.set_input("data", data)
    module.run()
    out =  module.get_output(0, tvm.nd.empty((1, 1000), 'float32', ctx=ctx)).asnumpy()

    ftimer = module.module.time_evaluator("run", ctx, number=REPEATS, repeat=num_iter)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
            (np.mean(prof_res), np.std(prof_res)))

def build_network():
    if RASP:
        target = tvm.target.arm_cpu("rasp3b")
    else:
        target = 'llvm'

    network = load_layers()
    params, dtypes = load_parameter()

    with autotvm.apply_history_best(log_file): 
        with nnvm.compiler.build_config(opt_level=opt_level):
            graph, lib, params = nnvm.compiler.build(network, target, 
                dtype=dtypes, 
                shape={"data":input_shape}, 
                params=params)

    return graph, lib, params


if __name__ == '__main__':
    get_network()
    print ("network")
    # Load existing NNVM module
    directory = '/sampa/home/cowanmeg/tvm-current/end2end/models'
    data_np, data = load_test_image()

    if RASP:
        lib_fname = os.path.join(directory, "rasp_vggnet.tar")
        loaded_json = open(os.path.join(directory, "rasp_vggnet.json")).read()
        loaded_params = bytearray(open(os.path.join(directory, "rasp_vggnet.params"), "rb").read())
        remote.upload(lib_fname)
        rlib = remote.load_module("rasp_vggnet.tar")
        module = graph_runtime.create(loaded_json, rlib, ctx)
        params = nnvm.compiler.load_param_dict(loaded_params)
    else:
        loaded_lib = tvm.module.load(os.path.join(directory, "vggnet.so"))
        loaded_json = open(os.path.join(directory, "vggnet.json")).read()
        loaded_params = bytearray(open(os.path.join(directory, "vggnet.params"), "rb").read())
        module = graph_runtime.create(loaded_json, loaded_lib, ctx)
        params = nnvm.compiler.load_param_dict(loaded_params)
        
    module.set_input(**params)
    run(data, 1, ctx)
 
