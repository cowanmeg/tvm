import os
import sys
import numpy as np
import tvm
import topi
import logging
import topi.testing
import nnvm.testing
import nnvm.compiler
from tvm import rpc, autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
from tvm.contrib import util
from tvm.autotvm.task.nnvm_integration import serialize_args
from tvm.contrib.util import tempdir
from end2end.nnvm_vgg import get_network
import tvm.contrib.graph_runtime as runtime


target = tvm.target.arm_cpu("rasp3b")
target_host = 'llvm -device=arm_cpu -target=arm-linux-gnueabihf -mattr=+neon'
device_key = 'rpi3b'
host = '10.77.1.69'
port = 9090
remote = rpc.connect(host, port)
ctx = remote.context(str(target), 0)
log_file =  os.environ["TVM_ROOT"] + '/tuner/logs/vgg11_rasp3b.log'
# log_file =  os.environ["TVM_ROOT"] + '/tuner/logs/empty.log'

def load_network():
     # compile kernels with history best records
    net, params, dtypes, input_shape, output_shape = get_network()
    print("Compile...")
    with autotvm.apply_history_best(log_file):
        with nnvm.compiler.build_config(opt_level=3):
            graph, lib, params = nnvm.compiler.build(
                net, target=target, target_host=target_host,
                shape={'data': input_shape}, params=params, dtype=dtypes)

    # export library
    tmp = tempdir()
    filename = "net.tar"
    lib.export_library(tmp.relpath(filename))

    # upload module to device
    print("Upload...")
    remote.upload(tmp.relpath(filename))
    rlib = remote.load_module(filename)

    # upload parameters to device
    module = runtime.create(graph, rlib, ctx)
    return module, params
    

def time(module, params):
    data_tvm = tvm.nd.array((np.random.uniform(size=(1, 224, 224, 3))).astype('float32'))
    module.set_input('data', data_tvm)
    module.set_input(**params)
    # evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=30)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
            (np.mean(prof_res), np.std(prof_res)))

def run(module, params, data):
    data_tvm = tvm.nd.array(data)
    module.set_input('data', data_tvm)
    module.set_input(**params)
    out =  module.get_output(0, tvm.nd.empty((1, 1000), 'float32', ctx=ctx)).asnumpy()
    return out


if __name__ == '__main__':
    module, params = load_network()
    time(module, params)
    run(module, params, np.random.uniform(size=(1, 224, 224, 3)).astype('float32'))