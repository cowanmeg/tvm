"""Benchmark script for ImageNet models on ARM CPU with low precision.
"""
import argparse
import os
import sys
import numpy as np
import tvm
from tvm.contrib.util import tempdir
import nnvm.compiler
import nnvm.testing
from tvm import autotvm, rpc
from  end2end.nnvm_vgg import get_network as get_vggnet
from end2end.fused_nnvm_vgg import get_network as get_fused_vggnet
from end2end.nnvm_vgg_bn import get_network as get_vgg_bn
from end2end.nnvm_vgg_sn_scale import get_network as get_vgg_sn
from end2end.nnvm_alexnet import get_network as get_alexnet

log_file =  os.environ["TVM_ROOT"] + '/tuner/logs/vggnet_rasp3b.log'

DEBUG=True

if DEBUG:
    from tvm.contrib.debugger import debug_runtime as runtime
    host = '10.77.1.69'
    port = 9090
    remote = rpc.connect(host, port) 
    ctx = remote.cpu(0)
else:
    import tvm.contrib.graph_runtime as runtime
    tracker = tvm.rpc.connect_tracker('fleet', 9190)
    remote = tracker.request('rpi3b')
    ctx = remote.context(str(target), 0)
       

def print_progress(msg):
    """print progress message
    
    Parameters
    ----------
    msg: str
        The message to print
    """
    sys.stdout.write(msg + "\r")
    sys.stdout.flush()

def evaluate_network(network, target, target_host, repeat):
    print_progress(network)
    if network == '0': # VGGNet with batch normalization, not tuned
        log_file =  os.environ["TVM_ROOT"] + '/tuner/logs/empty.log'
        net, dtypes, params, input_shape, output_shape = get_vgg_bn()
    elif network == '1': # VGGNet with batch normalization, tuned
        log_file =  os.environ["TVM_ROOT"] + '/tuner/logs/bn_vggnet_rasp3b.log'
        net, dtypes, params, input_shape, output_shape = get_vgg_bn()
    if network == '2': # VGGNet with fp weight scaling, shift normalization tuned
        log_file =  os.environ["TVM_ROOT"] + '/tuner/logs/sn_vggnet_rasp3b.log'
        net, dtypes, params, input_shape, output_shape = get_vgg_sn()
    elif network == 'vggnet': # VGGNet with shift normalization
        log_file =  os.environ["TVM_ROOT"] + '/tuner/logs/vggnet_rasp3b.log'
        net, dtypes, params, input_shape, output_shape = get_vggnet()
    elif network == '10':
        log_file =  os.environ["TVM_ROOT"] + '/tuner/logs/alexnet_rasp3b.log'
        net, dtypes, params, input_shape, output_shape = get_alexnet()
    else:
        log_file =  os.environ["TVM_ROOT"] + '/tuner/logs/fused_vggnet_rasp3b.log'
        net, dtypes, params, input_shape, output_shape = get_fused_vggnet()


    print_progress("%-20s building..." % network)
    with autotvm.apply_history_best(log_file): 
        with nnvm.compiler.build_config(opt_level=0):
            graph, lib, params = nnvm.compiler.build(
                net, target=target, target_host=target_host,
                shape={'data': input_shape}, params=params, dtype=dtypes)

    tmp = tempdir()
    if 'android' in str(target):
        from tvm.contrib import ndk
        filename = "%s.so" % network
        lib.export_library(tmp.relpath(filename), ndk.create_shared)
    else:
        filename = "%s.tar" % network
        lib.export_library(tmp.relpath(filename))

    # upload library and params
    print_progress("%-20s uploading..." % network)
    remote.upload(tmp.relpath(filename))

    rlib = remote.load_module(filename)
    module = runtime.create(graph, rlib, ctx)
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype('float32'))
    module.set_input('data', data_tvm)
    module.set_input(**params)
    
    # evaluate
    print_progress("%-20s evaluating..." % network)
    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=repeat)
    prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
    print("%-20s %-19s (%s)" % (network, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))

    if DEBUG:
        module.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, choices=
                        ['0', '1', '2', '10', 'vggnet', 'fused'],
                        help='The name of neural network')
    # parser.add_argument("--model", type=str, choices=
    #                     ['rk3399', 'mate10', 'mate10pro', 'p20', 'p20pro',
    #                      'pixel2', 'rasp3b', 'pynq'], default='rk3399',
    #                     help="The model of the test device. If your device is not listed in "
    #                          "the choices list, pick the most similar one as argument.")
    parser.add_argument("--host", type=str, default='fleet')
    parser.add_argument("--port", type=int, default=9190)
    parser.add_argument("--rpc-key", type=str, default='rpi3b')
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    # if args.network is None:
    #     networks = ['squeezenet_v1.1', 'mobilenet', 'resnet-18', 'vgg-16']
    # else:
    #     networks = [args.network]

    target = tvm.target.arm_cpu('rasp3b')
    target_host = None

    print("--------------------------------------------------")
    print("%-20s %-20s" % ("Network Name", "Mean Inference Time (std dev)"))
    print("--------------------------------------------------")
    evaluate_network(args.network, target, target_host, args.repeat)