"""Conversion script from Tensorflow to NNVM model for 2-bit, 1-bit VGGNet
"""

import numpy as np
import os
import sys
import tvm
import math
import json
import topi.testing
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list
from nnvm.compiler import graph_util
from nnvm.graph import GraphIndex
from PIL import Image
from tvm.contrib import util, rpc
from topi.nn.bitserial_conv2d import bitpack
from topi.util import get_const_tuple
from topi.nn.pad import pad
import pickle

from tvm.contrib import graph_runtime
# from tvm.contrib.debugger import debug_runtime as graph_runtime

import tensorflow as tf
from riptide.binary.binary_funcs import *
from riptide.binary.binary_layers import Config
from riptide.models.vggnet import vggnet
# tf.enable_eager_execution()


# Load tf model and weights
actQ = DQuantize
weightQ = XQuantize
bits = 2.0
use_act = False
use_bn = False
use_maxpool = True
# pure_shiftnorm = False
config = Config(actQ=actQ, weightQ=weightQ, bits=bits, use_act=use_act, use_bn=use_bn, 
                use_maxpool=use_maxpool)
graph = tf.Graph()
with graph.as_default():
    with config:
        model = vggnet(classes=1000)

    test_data = tf.keras.layers.Input(shape=[224, 224, 3], batch_size = 1)
    out_tensor = model(test_data, training=False)
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, '/shared/jwfromm/models/vggnet_medlr_2/model.ckpt-1201110') # imagenet
        input_shape = (1, 224, 224, 3)
        output_shape = (1, 1000)
        # print ("Tensorflow output:", get_numpy(sess, model(tf.ones(shape=input_shape), training=False)))

        # saver.restore(sess, '/shared/jwfromm/vgg_dorefa_shiftnorm_scalu/model.ckpt-60000') # cifar10
        # input_shape = (1, 32, 32, 3)
        # output_shape = (1, 10)
    # model.load_weights('/shared/jwfromm/models/vgg_dorefa_true_shiftnorm_confirm/model.ckpt-0')

RASP = True
REPEATS = 20
opt_level = 3
bitpack_weights = True 
save = False
directory = '/sampa/home/cowanmeg/tvm-current/end2end/models'
filename = directory + '/network.txt'
f = open(filename, 'w')

local_ctx = tvm.cpu(0)
if RASP:
    host = '10.77.1.69'
    port = 9090
    remote = rpc.connect(host, port)
    ctx = remote.cpu(0)
else:
    ctx = local_ctx

abits = 2
wbits = 1
layout = 'NHWC'
kernel_layout = 'HWIO'
kernel_pack_axis = 2
if RASP:
    kernel_bit_axis = 2
else:
    kernel_bit_axis = 4
kernel_pack_dtype = 'uint8'
output_dtype = 'float32'

def convert_same_padding(input_shape, output_shape, kernel, strides):
    def _get_pad_pair(i, o, k, s):
        pad = max((o - 1) * s + k - i, 0)

        pad_before = pad // 2
        pad_after = pad - pad_before
        return [pad_before, pad_after]

    pad_v = _get_pad_pair(input_shape[1], output_shape[1], kernel[0], strides[0])
    pad_h = _get_pad_pair(input_shape[2], output_shape[2], kernel[1], strides[1])
    return [pad_v[0], pad_h[0], pad_v[1], pad_h[1]]

def map_neg1_to_0(bp_weights):
    weights = np.copy(bp_weights)
    for x in np.nditer(weights, op_flags=['readwrite']):
        x[...] = 1 if x == 1 else 0
    return weights

# Loading in numpy parameters
def load_parameter(stop_layer=None):
    params = {}
    dtypes = {}

    # Load in trained parameters
    prev_layers = []
    for layer in model.layers:   
        if 'binary_conv2d' in layer.name:
            parameters = get_numpy(sess, get_quantize_bits(layer.weights[0]))
            shift = parameters[0]
            bp_weights = parameters[1].astype('int16')
            # shift is a power of 2 -> take log so it can be implementd as a right shift
            shift = -1 * np.log2(shift).flatten().astype('int16')
            key = layer.name + 'shift_shift'
            params[key] = shift
            dtypes[key] = 'int16'
            
            weights = map_neg1_to_0(bp_weights)
            key = layer.name + '_weight'
            if bitpack_weights:
                A = tvm.placeholder(weights.shape, dtype='int16')
                # Raspberry pi requires input channels to be multiple of 8
                if RASP and weights.shape[3] % 8 != 0:
                    CI_PAD = weights.shape[3] % 8
                    A = pad(A, [0, 0, 0, 0], [0, 0, 0, CI_PAD], name='PaddedWeights')
                B = bitpack(A, wbits, pack_axis=kernel_pack_axis, bit_axis=kernel_bit_axis, pack_type=kernel_pack_dtype)
                weights_bitpacked = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), local_ctx)
                s = tvm.create_schedule(B.op)
                func = tvm.build(s, [A, B], "llvm")
                weights_tvm = tvm.nd.array(weights, local_ctx)
                weights_bitpacked = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), local_ctx)
                func(weights_tvm, weights_bitpacked)
                params[key] = weights_bitpacked.asnumpy()
                dtypes[key] = kernel_pack_dtype
            else:
                params[key] = weights
                dtypes[key] = 'int16'

        elif 'conv2d' in layer.name:
            params[layer.name + '_weight'] = get_numpy(sess, layer.weights[0])
            dtypes[layer.name + '_weight'] = 'float32'

        elif 'shift_normalization' in layer.name:
            # Fuse this shift with the previous binary layer's shift
            prev_layer = prev_layers[-1]
            if 'max_pooling2d' in prev_layer.name:
                prev_layer = prev_layers[-2]

            approximate_std, quantized_means = get_numpy(sess, get_shiftnorm_ap2(layer, conv_weights=prev_layer.weights[0].value()))
            conv_shift = params[prev_layer.name + 'shift_shift']
            shift = -1 * np.log2(approximate_std).astype('int16')
            total_shift = conv_shift + shift
            params[prev_layer.name + 'shift_shift'] = total_shift
            mean = (quantized_means).astype('int16')
            r = 1 << (total_shift - 1)
            params[prev_layer.name + 'round'] = r - mean
            dtypes[prev_layer.name + 'round'] = 'int16'

            # For clip thresholds
            zeros = np.zeros(shape=shift.shape).astype('int16')
            bit_constant = int((2.0**abits) - 1)
            #  max int that can be represented (3 for 2 bit) but in pre-shifted space
            threshold = np.left_shift(bit_constant, total_shift).astype('int16')
            key = prev_layer.name + 'clip_a_min'
            params[key] = zeros
            dtypes[key] = 'int16'
            key = prev_layer.name + 'clip_a_max'
            params[key] = threshold
            dtypes[key] = 'int16'

        elif 'binary_dense' in layer.name:
            w = get_numpy(sess, get_quantize_bits(layer.weights[0]))
            shift = w[0]
            weights = w[1]
            # Convert shift so that it's int representing amount to right shift by
            shift = np.log2(1.0/shift).flatten().astype('int16')
            key = layer.name + 'shift_shift'
            params[key] = shift
            dtypes[key] = 'int16'
            # For clip thresholds
            zeros = np.zeros(shape=shift.shape).astype('int16')
            bit_constant = int((2.0**abits) - 1)
            threshold = np.left_shift(bit_constant, shift).astype('int16')
            key = layer.name + 'clip_a_min'
            params[key] = zeros
            dtypes[key] = 'int16'
            key = layer.name + 'clip_a_max'
            params[key] = threshold
            dtypes[key] = 'int16'

            # Map -1 weights to 0
            bp_weights = weights.astype('int16').transpose()
            weights = np.copy(bp_weights)
            for x in np.nditer(weights, op_flags=['readwrite']):
                x[...] = 1 if x == 1 else 0
            key = layer.name + '_weight'
            if bitpack_weights:
                A = tvm.placeholder(weights.shape, dtype='int16')
                B = bitpack(A, wbits, pack_axis=1, bit_axis=1, pack_type=kernel_pack_dtype)
                weights_bitpacked = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), local_ctx)
                s = tvm.create_schedule(B.op)
                func = tvm.build(s, [A, B], "llvm")
                weights_tvm = tvm.nd.array(weights, local_ctx)
                weights_bitpacked = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), local_ctx)
                func(weights_tvm, weights_bitpacked)
                params[key] = weights_bitpacked.asnumpy()
                dtypes[key] = kernel_pack_dtype
            else:
                params[key] = weights
                dtypes[key] = 'int16'

        elif 'batch_normalization' in layer.name:
            moving_mean, moving_var = get_numpy(sess, layer.weights)
            _scale = (1 / np.sqrt(moving_var + layer.epsilon)).astype('float32')
            _shift = (-1*moving_mean * _scale).astype('float32')

            params[layer.name +  '_scale'] = _scale
            params[layer.name + '_shift'] = _shift
            dtypes[layer.name +  '_scale'] = 'float32'
            dtypes[layer.name +  '_shift'] = 'float32'
        
        elif 'scalu' in layer.name:
            params[layer.name] = get_numpy(sess, layer.weights)
            dtypes[layer.name] = 'float32'

        prev_layers.append(layer)

        if stop_layer == layer.name:
            break

    for k, v in params.items():
        print (k, v.shape)
    return params, dtypes

def load_layers(stop_layer=None):
    network = sym.Variable(name="data")

    # Load in trained parameters
    for layer in model.layers:
        if "binary_conv2d" in layer.name:
            if (layer.padding in 'same'):
                padding = convert_same_padding(layer.input_shape, layer.output_shape, layer.kernel_size, layer.strides)
            else:
                padding = (0, 0)

            if bitpack_weights:
                bit_axis = kernel_bit_axis
            else:
                bit_axis = -1
            network = sym.bitserial_conv2d(data=network, kernel_size=layer.kernel_size, channels=layer.filters,
                            padding=padding, strides=layer.strides,
                            layout=layout, kernel_layout=kernel_layout, use_pool=False,
                            activation_bits=abits, weight_bits=wbits, pack_dtype='uint8',
                            out_dtype='int16', bit_axis=bit_axis, name=layer.name)
            # Scaling back down to quantized data
            # Step 1 -> Add to imitate round nearest instead of floor
            
            r = sym.Variable(layer.name + 'round', shape=layer.output_shape)
            network = network + r
            network = sym.clip_channelwise(network, axis=3, name=layer.name+'clip')
            network = sym.right_shift_channelwise(network, axis=3, name=layer.name+'shift')
            if save:
                print('network = sym.bitserial_conv2d(data=network, ' \
                    'kernel_size=%s, channels=%d, padding=%s, strides=%s, '\
                    'layout=layout, kernel_layout=kernel_layout, use_pool=False, ' \
                    'activation_bits=abits, weight_bits=wbits, pack_dtype=\'uint8\', ' \
                    'out_dtype=\'int16\', bit_axis=bit_axis, name=\'%s\')' % (layer.kernel_size, layer.filters, padding, layer.strides, layer.name))
                print('r = sym.Variable(\'%sround\', shape=%s)' % (layer.name, layer.output_shape))
                print ('network = network + r')
                print ('sym.clip_channelwise(network, axis=3, name=\'%s\' +\'clip\')' % layer.name)
                print ('network = sym.right_shift_channelwise(network, axis=3, name= \'%s\'+\'shift\')' % layer.name)
        elif 'conv2d' in layer.name:
            # This is the first layer - floating point conv
            channels = layer.filters
            kernel_size = layer.kernel_size
            strides = layer.strides
            if  (layer.padding in 'same'):
                padding = convert_same_padding(layer.input_shape, layer.output_shape, layer.kernel_size, layer.strides)
            else:
                padding = (0, 0)
            network = sym.conv2d(data=network, channels=channels, kernel_size=kernel_size, strides=strides, padding=padding, 
                        layout=layout, kernel_layout=kernel_layout, use_bias=layer.use_bias, name=layer.name)
            if save:
                print ('network = sym.conv2d(data=network, channels=%d, kernel_size=%s, strides=%s, padding=%s,\
                layout=layout, kernel_layout=kernel_layout, use_bias=%s, name=\'%s\')' % (channels, kernel_size, strides, padding, layer.use_bias, layer.name))
        elif 'average_pooling2d' in layer.name:
            network = sym.global_avg_pool2d(network, name=layer.name, layout=layout)
            if save:
                print ('network = sym.global_avg_pool2d(network, name=\'%s\', layout=layout)' % layer.name)
        elif 'max_pooling2d' in layer.name:
            network = sym.max_pool2d(network, name=layer.name, layout=layout, pool_size=layer.pool_size, padding=(0, 0), strides=layer.strides)
            if save:
                print ('network = sym.max_pool2d(network, name=\'%s\', layout=layout, pool_size=%s, padding=(0, 0), strides=%s)' % (layer.name, layer.pool_size, layer.strides))
        elif 'binary_dense' in layer.name:
            network = sym.flatten(data=network)
            network = sym.bitserial_dense(data=network, units=layer.units, activation_bits=abits, weight_bits=wbits,
                pack_dtype='uint8', out_dtype='int16', name=layer.name)

            # shift norm
            r = sym.Variable(layer.name + 'round', shape=layer.output_shape)
            network = network + r
            # TODO: temp to get this working
            network = sym.clip_channelwise(network, axis=0, name=layer.name+'clip')
            network = sym.right_shift_channelwise(network, axis=0, name=layer.name+'shift')
            if save:
                print ('network = sym.flatten(data=network)')
                print ('network = sym.bitserial_dense(data=network, units=%d, activation_bits=abits, weight_bits=wbits, ' \
                    'pack_dtype=\'uint8\', out_dtype=\'int16\', name=\'%s\')' % (layer.units, layer.name))
                print ('r = sym.Variable(\'%sround\', shape=%s)' % (layer.name, layer.output_shape))
                print ('network = network + r')
                print ('network = sym.clip_channelwise(network, axis=0, name=\'%s\'+\'clip\')' % layer.name)
                print ('network = sym.right_shift_channelwise(network, axis=0, name=\'%s\'+\'shift\')' % layer.name)
        elif 'scale' in layer.name:
            network = network * layer.scale
            # Quantize
            network = sym.clip(data=network, a_min=0.0, a_max=1.0)
            data = network * ((1 << abits) - 1) + 0.5
            network = sym.cast(data=data, dtype='int16')
            if save:
                print ('network = network * %d' % layer.scale)
                print ('network = sym.clip(data=network, a_min=0.0, a_max=1.0)')
                print ('data = network * ((1 << abits) - 1) + 0.5')
                print ('network = sym.cast(data=data, dtype=\'int16\')')
        elif 'scalu' in layer.name:
            scale = sym.Variable(layer.name, shape=layer.input_shape)
            network = sym.cast(data=network, dtype='float32')
            network = network * scale
            if save:
                print ('scale = sym.Variable(\'%s\', shape=%s)' % (layer.name, layer.input_shape))
                print ('network = sym.cast(data=network, dtype=\'float32\')')
                print ('network = network * scale')
        elif 'batch_normalization' in layer.name:
            # Currently NNVM doesn't support NHWC batch norm - computing it from the components
            moving_mean, moving_var = get_numpy(sess, layer.weights)
            scale = sym.Variable(layer.name + '_scale', shape=moving_var.shape)
            shift = sym.Variable(layer.name + '_shift', shape=moving_mean.shape)
            network = sym.broadcast_add(sym.broadcast_mul(network, scale), shift)
            if save:
                print ('scale = sym.Variable(\'%s_scale\', shape=%s)' % (layer.name, moving_var.shape))
                print ('shift = sym.Variable(\'%s_shift\', shape=%s)' % (layer.name, moving_mean.shape))
                print('network = sym.broadcast_add(sym.broadcast_mul(network, scale), shift)')

        if stop_layer == layer.name:
            global output_shape
            output_shape = layer.output_shape
            global output_dtype
            if 'scalu' in stop_layer or 'conv2d' == stop_layer:
                output_dtype = 'float32'
            else:
                output_dtype = 'int16'
            break
    return network
  

def load_test_image():
    data_np = np.ones(shape=(input_shape)).astype('float32')
    data_tvm = tvm.nd.array(data_np)
    return data_np, data_tvm

def run(data, stop_layer=None):
    graph, lib, params, graph_index, graph_json = build_network(stop_layer)

    if RASP:
        tmp = util.tempdir()
        lib_fname = tmp.relpath('net.o')
        lib.save(lib_fname)
        remote.upload(lib_fname)

        rlib = remote.load_module('net.o')
        rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}
        module = graph_runtime.create(graph, rlib, ctx)
        # module = graph_runtime.create(graph, rlib, ctx, dump_root="/tmp/tvmdbg")
    else:
        rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}
        module = graph_runtime.create(graph, lib, ctx)
        
    module.set_input(**rparams)
    module.set_input("data", data)
    module.run()
    out =  module.get_output(0, tvm.nd.empty(output_shape, output_dtype, ctx=ctx)).asnumpy()
    
    ftimer = module.module.time_evaluator("run", ctx, 3)
    print (ftimer())
    return out

def get_network():
    network = load_layers()
    params, dtypes = load_parameter()
    return network, params, dtypes, input_shape, output_shape

def build_network(stop_layer=None):
    if RASP:
        target = tvm.target.arm_cpu("rasp3b")
    else:
        target = 'llvm'

    network = load_layers(stop_layer)
    params, dtypes = load_parameter(stop_layer)

    # Does it make a difference if I call with tvm.target.rasp() here?
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(network, target, 
            dtype=dtypes, 
            shape={"data":input_shape}, 
            params=params)
        # print (graph.ir())
    # print (graph.json())
    graph_json = json.loads(graph.json())
    graph_index = GraphIndex(graph)

    if save:
        pickle.dump(dtypes, open(os.path.join(directory, "vggnet_dtypes.p"), 'wb'))
        if not os.path.exists(directory):
            os.makedirs(directory)
        home_dir = util.tempdir()
        path_lib = os.path.join(directory, "rasp_vggnet.tar")
        lib.export_library(path_lib)
        with open(os.path.join(directory, "rasp_vggnet.json"), "w") as fo:
            fo.write(graph.json())
        with open(os.path.join(directory, "rasp_vggnet.params"), "wb") as fo:
            fo.write(nnvm.compiler.save_param_dict(params))

    return graph, lib, params, graph_index, graph_json


if __name__ == '__main__':
    with graph.as_default():
        data_tvm, data_tf = load_test_image()
        out = run(data_tvm)
