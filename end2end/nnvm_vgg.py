import numpy as np
import os
import tvm
import math
from tvm.contrib import graph_runtime
import topi.testing
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list
from nnvm.compiler import graph_util
from tvm.contrib import graph_runtime as runtime
from PIL import Image
from tvm.contrib import util, rpc

import tensorflow as tf
from riptide.binary.binary_funcs import *
from riptide.binary.binary_layers import Config
from riptide.models.vgg11 import vgg11
# tf.enable_eager_execution()


# Load tf model and weights
actQ = DQuantize
weightQ = XQuantize
bits = 2.0#load_clusters(2)
use_act = False
use_bn = False
use_maxpool = False
pure_shiftnorm = True
config = Config(actQ=actQ, weightQ=weightQ, bits=bits, use_act=use_act, use_bn=use_bn, use_maxpool=use_maxpool, pure_shiftnorm=pure_shiftnorm)

with config:
    model = vgg11(classes=10)

test_data = tf.keras.layers.Input(shape=[32, 32, 3], batch_size = 1)
out_tensor = model(test_data, training=False)
sess = tf.Session()
with sess.as_default():
    saver = tf.train.Saver()
    saver.restore(sess, '/shared/jwfromm/models/vgg_dorefa_true_shiftnorm_confirm/model.ckpt-60000')
# model.load_weights('/shared/jwfromm/models/vgg_dorefa_true_shiftnorm_confirm/model.ckpt-0')

RASP = True
opt_level = 0

abits=2
wbits=1
layout = 'NHWC'
kernel_layout = 'HWIO'
data_shape = (1, 32, 32, 3)
out_shape = (1, 10)

def convert_same_padding(input_shape, output_shape, kernel, strides):
    def _get_pad_pair(i, o, k, s):
        pad = max((o - 1) * s + k - i, 0)
        # if i % s == 0:
        #     pad = max(k - s, 0)
        # else:
        #     pad = max(k - (i % s), 0)
        pad_before = pad // 2
        pad_after = pad - pad_before
        return [pad_before, pad_after]

    pad_v = _get_pad_pair(input_shape[1], output_shape[1], kernel[0], strides[0])
    pad_h = _get_pad_pair(input_shape[2], output_shape[2], kernel[1], strides[1])

    return [pad_v[0], pad_h[0], pad_v[1], pad_h[1]]

# Loading in numpy parameters
def load_parameter():
    params = {}
    dtypes = {}

    # Load in trained parameters
    prev_layer_name = None
    for layer in model.layers:        
        if 'binary_conv2d' in layer.name:
            w = get_numpy(sess, get_quantize_bits(layer.weights[0]))[0]
            shift = w[0]
            weights = w[1]
            # Convert shift so that it's int representing amount to right shift by
            shift = np.log2(1.0 / shift).flatten().astype('int16')
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
            bp_weights = weights.astype('int16')
            weights = np.copy(bp_weights)
            for x in np.nditer(weights, op_flags=['readwrite']):
                x[...] = 1 if x == 1 else 0
            key = layer.name + '_weight'
            params[key] = weights
            dtypes[key] = 'int16'
        elif 'conv2d' in layer.name:
            params[layer.name + '_weight'] = get_numpy(sess, layer.weights)[0]
            dtypes[layer.name + '_weight'] = 'float32'
            params[layer.name + '_bias'] = get_numpy(sess, layer.weights)[1]
            dtypes[layer.name + '_bias'] = 'float32'
        elif 'shift_normalization' in layer.name:
            # Fuse this shift with the previous binary layer's shift
            x = get_shiftnorm_ap2(sess, layer)
            shift = np.log2(x).flatten().astype('int16')
            params[prev_layer_name + 'shift_shift'] += shift

            params[prev_layer_name + 'round'] = 1 << (params[prev_layer_name + 'shift_shift'] - 1)
            dtypes[prev_layer_name + 'round'] = 'int16'
        elif 'binary_dense' in layer.name:
            w = get_numpy(sess, get_quantize_bits(layer.weights[0]))[0]
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
            params[key] = weights
            dtypes[key] = 'int16'

        elif 'batch_normalization' in layer.name:
            gamma, beta, moving_mean, moving_var = get_numpy(sess, layer.weights)
            # gamma = sym.Variable(layer.name + '_gamma', shape=g.shape)
            # beta = sym.Variable(layer.name + '_beta', shape=b.shape)
            # moving_mean = sym.Variable(layer.name + '_moving_mean', shape=m_m.shape)
            # moving_var = sym.Variable(layer.name + '_moving_var', shape=m_v.shape)
            _scale = (1 / np.sqrt(moving_var + layer.epsilon) * gamma).astype('float32')
            _shift = (-1*moving_mean * _scale + beta).astype('float32')

            params[layer.name +  '_scale'] = _scale
            params[layer.name + '_shift'] = _shift
            dtypes[layer.name +  '_scale'] = 'float32'
            dtypes[layer.name +  '_shift'] = 'float32'
        
        prev_layer_name = layer.name
    # for key in params:
    #         print (key, params[key].shape, dtypes[key])

    return params, dtypes

def load_network():
    def remove_prefix(text, prefix):
        if text.startswith(prefix):
            return text[len(prefix):]
        return text 

    network = sym.Variable(name="data")
    params = {}
    dtypes = {}
    input_shape = (32, 32)

    # Load in trained parameters
    for layer in model.layers:
        # print (layer.name, 'input:', layer.input_shape, 'output:', layer.output_shape)
        if "binary_conv2d" in layer.name:
            if (layer.padding in 'same'):
                padding = convert_same_padding(layer.input_shape, layer.output_shape, layer.kernel_size, layer.strides)
            else:
                padding = (0, 0)

            network = sym.bitserial_conv2d(data=network, kernel_size=layer.kernel_size, channels=layer.filters,
                            padding=padding, strides=layer.strides,
                            layout=layout, kernel_layout=kernel_layout, use_bias=False, 
                            activation_bits=abits, weight_bits=wbits,
                            out_dtype='int16', name=layer.name)
            # Scaling back down to quantized data
            # Step 1 -> Add to imitate round nearest instead of floor
            r = sym.Variable(layer.name + 'round', shape=layer.output_shape)
            network = network + r
            network = sym.clip_channelwise(network, axis=3, name=layer.name+'clip')
            network = sym.right_shift_channelwise(network, axis=3, name=layer.name+'shift')

        elif 'conv2d' in layer.name:
            channels = layer.filters
            kernel_size = layer.kernel_size
            strides = layer.strides
            if  (layer.padding in 'same'):
                padding = convert_same_padding(layer.input_shape, layer.output_shape, layer.kernel_size, layer.strides)
            else:
                padding = (0, 0)
            conv = sym.conv2d(data=network, channels=channels, kernel_size=kernel_size, strides=strides, padding=padding, 
                        layout=layout, kernel_layout=kernel_layout, use_bias=True, name=layer.name)
            network = sym.relu(data=conv)
            
        elif 'average_pooling2d' in layer.name:
            network = sym.global_avg_pool2d(network, name=layer.name, layout=layout)

        elif 'binary_dense' in layer.name:
            network = sym.flatten(data=network)
            network = sym.bitserial_dense(data=network, units=layer.units, activation_bits=abits, weight_bits=wbits,
                out_dtype='int16', name=layer.name)

        elif 'scale' in layer.name:
            network = network * layer.scale
            # Quantize
            network = sym.clip(data=network, a_min = 0.0, a_max = 1.0)
            data = network * ((1 << abits) - 1) + 0.5
            network = sym.cast(data=data, dtype='int16')
          
        elif 'shift_normalization' in layer.name:
            # Merge shift normalization with the proceeding binary operation
            continue 

        elif 'max_pooling2d' in layer.name:
            network = sym.max_pool2d(data=network, pool_size=layer.pool_size, strides=layer.strides, name=layer.name, layout=layout)
            
            

            
        elif 'batch_normalization' in layer.name:
            # Currently NNVM doesn't support NHWC batch norm - computing it from the components
            gamma, beta, moving_mean, moving_var = get_numpy(sess, layer.weights)
            scale = sym.Variable(layer.name + '_scale', shape=moving_var.shape)
            shift = sym.Variable(layer.name + '_shift', shape=moving_mean.shape)
            network = sym.broadcast_add(sym.broadcast_mul(network, scale), shift)
        else:
            print ("\t Didn't handle", layer.name)


    return network

def load_test_image():
    data_np = np.ones(shape=(1, 32, 32, 3)).astype('float32')
    data_tvm = tvm.nd.array(data_np)
    return data_np, data_tvm

def run(num_iter, ctx):
    module.set_input("data", data)
    module.run()
    print (out_shape)
    out =  module.get_output(0, tvm.nd.empty(out_shape, 'int16', ctx=ctx)).asnumpy()

    # check for no nans
    # print ("Any nan?", np.isnan(np.min(out)))
    # print (np.unique(out, return_counts=True))
    print ("output", out)

    ftimer = module.module.time_evaluator("run", ctx, num_iter)
    print (ftimer())

network = load_network()
params, dtypes = load_parameter()
data_np, data = load_test_image()
print (get_numpy(sess, model(tf.convert_to_tensor(data_np))))

if RASP:
    target = tvm.target.arm_cpu("rasp3b")
else:
    target = 'llvm'

with nnvm.compiler.build_config(opt_level=opt_level):
    graph, lib, params = nnvm.compiler.build(network, target, 
        dtype=dtypes, 
        shape={"data":data.shape}, 
        params=params)
    # print (graph.ir())

if RASP:
    host = '10.77.1.69'
    port = 9090

    tmp = util.tempdir()
    lib_fname = tmp.relpath('net.o')
    lib.save(lib_fname)

    remote = rpc.connect(host, port)
    remote.upload(lib_fname)

    ctx = remote.cpu(0)
    rlib = remote.load_module('net.o')
    rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}
    module = runtime.create(graph, rlib, ctx)
else:
    ctx = tvm.cpu(0)
    rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}
    module = runtime.create(graph, lib, ctx)
module.set_input(**rparams)
run(5, ctx)


