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

# log_file =  os.environ["TVM_ROOT"] + '/tuner/logs/vggnet_rasp3b.log'

# Network settings
abits = 2
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
    network = sym.Variable(name='data')
    network = sym.conv2d(data=network, channels=96, kernel_size=(7, 7), strides=(2, 2), padding=[2, 2, 3, 3], layout=layout, kernel_layout=kernel_layout, use_bias=False, name='conv2d')
    network = sym.max_pool2d(network, name='max_pooling2d', layout=layout, pool_size=(2, 2), padding=(0, 0), strides=(2, 2))
    scale = sym.Variable('batch_normalization_scale', shape=(96,))
    shift = sym.Variable('batch_normalization_shift', shape=(96,))
    network = sym.broadcast_add(sym.broadcast_mul(network, scale), shift)
    network = network * 1
    network = sym.clip(data=network, a_min=0.0, a_max=1.0)
    data = network * ((1 << abits) - 1) + 0.5
    network = sym.cast(data=data, dtype='int16')
    network = sym.bitserial_conv2d(data=network, kernel_size=(3, 3), channels=256, padding=[1, 1, 1, 1], strides=(1, 1), layout=layout, kernel_layout=kernel_layout, use_pool=False, activation_bits=abits, weight_bits=wbits, pack_dtype='uint8', out_dtype='int16', bit_axis=bit_axis, name='binary_conv2d')
    r = sym.Variable('binary_conv2dround', shape=(256, ))
    network = network + r
    sym.clip_channelwise(network, axis=3, name='binary_conv2d' +'clip')
    network = sym.right_shift_channelwise(network, axis=3, name= 'binary_conv2d'+'shift')
    network = sym.bitserial_conv2d(data=network, kernel_size=(3, 3), channels=256, padding=[1, 1, 1, 1], strides=(1, 1), layout=layout, kernel_layout=kernel_layout, use_pool=False, activation_bits=abits, weight_bits=wbits, pack_dtype='uint8', out_dtype='int16', bit_axis=bit_axis, name='binary_conv2d_1')
    r = sym.Variable('binary_conv2d_1round', shape=(256, ))
    network = network + r
    sym.clip_channelwise(network, axis=3, name='binary_conv2d_1' +'clip')
    network = sym.right_shift_channelwise(network, axis=3, name= 'binary_conv2d_1'+'shift')
    network = sym.bitserial_conv2d(data=network, kernel_size=(3, 3), channels=256, padding=[1, 1, 1, 1], strides=(1, 1), layout=layout, kernel_layout=kernel_layout, use_pool=False, activation_bits=abits, weight_bits=wbits, pack_dtype='uint8', out_dtype='int16', bit_axis=bit_axis, name='binary_conv2d_2')
    r = sym.Variable('binary_conv2d_2round', shape=(256, ))
    network = network + r
    sym.clip_channelwise(network, axis=3, name='binary_conv2d_2' +'clip')
    network = sym.right_shift_channelwise(network, axis=3, name= 'binary_conv2d_2'+'shift')
    network = sym.max_pool2d(network, name='max_pooling2d_1', layout=layout, pool_size=(2, 2), padding=(0, 0), strides=(2, 2))
    network = sym.bitserial_conv2d(data=network, kernel_size=(3, 3), channels=512, padding=[1, 1, 1, 1], strides=(1, 1), layout=layout, kernel_layout=kernel_layout, use_pool=False, activation_bits=abits, weight_bits=wbits, pack_dtype='uint8', out_dtype='int16', bit_axis=bit_axis, name='binary_conv2d_3')
    r = sym.Variable('binary_conv2d_3round', shape=(512, ))
    network = network + r
    sym.clip_channelwise(network, axis=3, name='binary_conv2d_3' +'clip')
    network = sym.right_shift_channelwise(network, axis=3, name= 'binary_conv2d_3'+'shift')
    network = sym.bitserial_conv2d(data=network, kernel_size=(3, 3), channels=512, padding=[1, 1, 1, 1], strides=(1, 1), layout=layout, kernel_layout=kernel_layout, use_pool=False, activation_bits=abits, weight_bits=wbits, pack_dtype='uint8', out_dtype='int16', bit_axis=bit_axis, name='binary_conv2d_4')
    r = sym.Variable('binary_conv2d_4round', shape=(512, ))
    network = network + r
    sym.clip_channelwise(network, axis=3, name='binary_conv2d_4' +'clip')
    network = sym.right_shift_channelwise(network, axis=3, name= 'binary_conv2d_4'+'shift')
    network = sym.bitserial_conv2d(data=network, kernel_size=(3, 3), channels=512, padding=[1, 1, 1, 1], strides=(1, 1), layout=layout, kernel_layout=kernel_layout, use_pool=False, activation_bits=abits, weight_bits=wbits, pack_dtype='uint8', out_dtype='int16', bit_axis=bit_axis, name='binary_conv2d_5')
    r = sym.Variable('binary_conv2d_5round', shape=(512, ))
    network = network + r
    sym.clip_channelwise(network, axis=3, name='binary_conv2d_5' +'clip')
    network = sym.right_shift_channelwise(network, axis=3, name= 'binary_conv2d_5'+'shift')
    network = sym.max_pool2d(network, name='max_pooling2d_2', layout=layout, pool_size=(2, 2), padding=(0, 0), strides=(2, 2))
    network = sym.bitserial_conv2d(data=network, kernel_size=(3, 3), channels=512, padding=[1, 1, 1, 1], strides=(1, 1), layout=layout, kernel_layout=kernel_layout, use_pool=False, activation_bits=abits, weight_bits=wbits, pack_dtype='uint8', out_dtype='int16', bit_axis=bit_axis, name='binary_conv2d_6')
    r = sym.Variable('binary_conv2d_6round', shape=(512, ))
    network = network + r
    sym.clip_channelwise(network, axis=3, name='binary_conv2d_6' +'clip')
    network = sym.right_shift_channelwise(network, axis=3, name= 'binary_conv2d_6'+'shift')
    network = sym.bitserial_conv2d(data=network, kernel_size=(3, 3), channels=512, padding=[1, 1, 1, 1], strides=(1, 1), layout=layout, kernel_layout=kernel_layout, use_pool=False, activation_bits=abits, weight_bits=wbits, pack_dtype='uint8', out_dtype='int16', bit_axis=bit_axis, name='binary_conv2d_7')
    r = sym.Variable('binary_conv2d_7round', shape=(512, ))
    network = network + r
    sym.clip_channelwise(network, axis=3, name='binary_conv2d_7' +'clip')
    network = sym.right_shift_channelwise(network, axis=3, name= 'binary_conv2d_7'+'shift')
    network = sym.bitserial_conv2d(data=network, kernel_size=(3, 3), channels=512, padding=[1, 1, 1, 1], strides=(1, 1), layout=layout, kernel_layout=kernel_layout, use_pool=False, activation_bits=abits, weight_bits=wbits, pack_dtype='uint8', out_dtype='int16', bit_axis=bit_axis, name='binary_conv2d_8')
    r = sym.Variable('binary_conv2d_8round', shape=(512, ))
    network = network + r
    sym.clip_channelwise(network, axis=3, name='binary_conv2d_8' +'clip')
    network = sym.right_shift_channelwise(network, axis=3, name= 'binary_conv2d_8'+'shift')
    network = sym.max_pool2d(network, name='max_pooling2d_3', layout=layout, pool_size=(2, 2), padding=(0, 0), strides=(2, 2))
    network = sym.flatten(data=network)
    network = sym.bitserial_dense(data=network, units=4096, activation_bits=abits, weight_bits=wbits, pack_dtype='uint8', out_dtype='int16', name='binary_dense')
    r = sym.Variable('binary_denseround', shape=(4096, ))
    network = network + r
    network = sym.clip_channelwise(network, axis=0, name='binary_dense'+'clip')
    network = sym.right_shift_channelwise(network, axis=0, name='binary_dense'+'shift')
    network = sym.flatten(data=network)
    network = sym.bitserial_dense(data=network, units=4096, activation_bits=abits, weight_bits=wbits, pack_dtype='uint8', out_dtype='int16', name='binary_dense_1')
    r = sym.Variable('binary_dense_1round', shape=(4096, ))
    network = network + r
    network = sym.clip_channelwise(network, axis=0, name='binary_dense_1'+'clip')
    network = sym.right_shift_channelwise(network, axis=0, name='binary_dense_1'+'shift')
    network = sym.flatten(data=network)
    network = sym.bitserial_dense(data=network, units=1000, activation_bits=abits, weight_bits=wbits, pack_dtype='uint8', out_dtype='int16', name='binary_dense_2')
    r = sym.Variable('binary_dense_2round', shape=(1000, ))
    network = network + r
    network = sym.clip_channelwise(network, axis=0, name='binary_dense_2'+'clip')
    network = sym.right_shift_channelwise(network, axis=0, name='binary_dense_2'+'shift')
    scale = sym.Variable('scalu', shape=(1, 1000))
    network = sym.cast(data=network, dtype='float32')
    network = network * scale

    d = '/sampa/home/cowanmeg/tvm-current/end2end/models/'
    dtypes = pickle.load(open(os.path.join(d, "vggnet_dtypes.p"), 'rb'))

    loaded_params = bytearray(open(os.path.join(d, "rasp_vggnet.params"), "rb").read())
    params = nnvm.compiler.load_param_dict(loaded_params)

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
    #if RASP:
    #    target = tvm.target.arm_cpu("rasp3b")
    #else:
    #    target = 'llvm'
    
    # Building network for the target
    target = 'llvm -target=arm-linux-gnueabihf --system-lib'
    opt_level = 3
    log_file = 'models/vggnet_rasp3b.txt' 
    network, dtypes, params, input_shape, output_shape  = get_network()

    with autotvm.apply_history_best(log_file): 
        with nnvm.compiler.build_config(opt_level=opt_level):
            graph, lib, params = nnvm.compiler.build(network, target, 
                dtype=dtypes, 
                shape={"data":input_shape}, 
                params=params)

    # Saving network library
    lib.save('models/vggnet.o')
    return graph, lib, params


if __name__ == '__main__':
    build_network() 
