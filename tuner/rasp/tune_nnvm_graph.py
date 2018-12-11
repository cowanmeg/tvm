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
from tvm.contrib.debugger import debug_runtime as debug_runtime

DEFAULT_TRIALS = 10
DEBUG = False # Can't debug remote graph
target = tvm.target.arm_cpu("rasp3b")
target_host = 'llvm -device=arm_cpu -target=arm-linux-gnueabihf -mattr=+neon'
device_key = 'rpi3b'
# log_file =  os.environ["TVM_ROOT"] + '/tuner/logs/vgg11_rasp3b.log'
log_file =  os.environ["TVM_ROOT"] + '/tuner/logs/empty.log'

tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'early_stopping': 400,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(
            build_func='default'),
        runner=autotvm.RPCRunner(
            device_key, host='fleet', port=9190,
            number=5,
            timeout=5,
        ),
    ),
}


def measure_best(net, params, dtypes, input_shape, output_shape):
     # compile kernels with history best records
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
    remote = autotvm.measure.request_remote(device_key, 'fleet', 9190,
                                            timeout=10000)
    remote.upload(tmp.relpath(filename))
    rlib = remote.load_module(filename)

    # upload parameters to device
    ctx = remote.context(str(target), 0)
    if DEBUG:
        module = debug_runtime.create(graph, rlib, ctx, dump_root="/tmp/tvmdbg")
    else:
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

def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=10,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                    early_stopping=early_stopping,
                    measure_option=measure_option,
                    callbacks=[
                        autotvm.callback.progress_bar(n_trial, prefix=prefix),
                        autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    # os.remove(tmp_log_file)

def tune_and_evaluate(trials, tune):
    with target:
        net, params, dtypes, input_shape, output_shape = get_network()
        tasks = autotvm.task.extract_from_graph(net, target=target,
                                            shape={'data': input_shape}, dtype=dtypes,
                                            symbols=(nnvm.sym.bitserial_conv2d, nnvm.sym.bitserial_dense))
       
    # run tuning tasks
    if tune:
        print("Tuning...")
        tune_tasks(tasks, **tuning_option, n_trial=trials)
    measure_best(net, params, dtypes, input_shape, output_shape)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        tune = True
        trials = int(sys.argv[1])
    else:
        tune = False
        trials = DEFAULT_TRIALS
    tune_and_evaluate(trials, tune)
   