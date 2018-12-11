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
log_file =  os.environ["TVM_ROOT"] + '/tuner/logs/vgg11_rasp3b.log'


def measure_best():
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
    host = '10.77.1.69'
    port = 9090
    remote = rpc.connect(host, port)

    remote.upload(tmp.relpath(filename))
    rlib = remote.load_module(filename)

    # upload parameters to device
    ctx = remote.context(str(target), 0)
    module = runtime.create(graph, rlib, ctx)
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype('float32'))
    module.set_input('data', data_tvm)
    module.set_input(**params)

    # evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=30)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
            (np.mean(prof_res), np.std(prof_res)))


if __name__ == '__main__':
    measure_best()