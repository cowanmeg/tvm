"""Example code to do bitserial dense."""
import os
import sys
import numpy as np
import tvm
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
from tvm.contrib import rpc, util
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.task.nnvm_integration import serialize_args

DEFAULT_TRIALS = 10
target = tvm.target.create('llvm -device=arm_cpu -target=arm-linux-gnueabihf -mattr=+neon')
device_key = 'rpi3b'
log_file =  os.environ["TVM_ROOT"] + '/tuner/logs/bitserial_dense_rasp3b.log'

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

def generate_quantized_np(shape, bits, out_dtype):
    np.random.seed(0)
    min_val = 0
    max_val = 1 << bits
    return np.random.randint(min_val, max_val, size=shape).astype(out_dtype)

def verify_bitserial_dense(batch, in_dim, out_dim, data_bits, weight_bits, in_dtype, pack_dtype, out_dtype, dorefa):
    with autotvm.apply_history_best(log_file):
        with tvm.target.arm_cpu('rasp3b'):
            A = tvm.placeholder((batch, in_dim), dtype=in_dtype, name='A')
            W = tvm.placeholder((out_dim, in_dim), dtype=in_dtype, name='W')
            B = topi.nn.bitserial_dense(A, W, data_bits, weight_bits, pack_dtype, out_dtype, dorefa)
            s = topi.generic.schedule_bitserial_dense(B)
            # s = tvm.create_schedule(B.op)
            # print (tvm.lower(s, [A, W, B], simple_mode=True))

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)

    def get_ref_data():
        a_np = generate_quantized_np(get_const_tuple(A.shape), data_bits, in_dtype)
        w_np = generate_quantized_np(get_const_tuple(W.shape), weight_bits, in_dtype)
        w_ = np.copy(w_np).astype(out_dtype)
        for x in np.nditer(w_, op_flags=['readwrite']):
            x[...] = 1 if x == 1 else -1
        b_np = np.dot(a_np.astype(out_dtype), w_.T.astype(out_dtype))
        return a_np, w_np, b_np
    a_np, w_np, b_np = get_ref_data()

    remote = autotvm.measure.request_remote("rasp3b", 'localhost', 9190, timeout=10)
    ctx = remote.context(str(target), 0)
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func = tvm.build(s, [A, W, B], target)

     # upload to rpi
    temp = util.tempdir()
    path = temp.relpath('dense.o')
    func.save(path)
    remote.upload(path)
    func = remote.load_module('dense.o')

    func(a, w, b)
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

def tune_tasks(tsk,
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

    # for i, tsk in enumerate(reversed(tasks)):
    # prefix = "[Task %2d/%2d] " % (i+1, len(tasks))
    prefix = '[Task 1/1]'

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

def tune_and_evaluate(batch, in_dim, out_dim, activation_bits, weight_bits, in_dtype, pack_dtype, out_dtype, dorefa, trials):

    A = tvm.placeholder((batch, in_dim), dtype=in_dtype, name='A')
    W = tvm.placeholder((out_dim, in_dim), dtype=in_dtype, name='W')
    args = [A, W, activation_bits, weight_bits, pack_dtype, out_dtype, dorefa]
    args = serialize_args(args)
    task = autotvm.task.create("topi_arm_cpu_bitserial_dense",
                    args=args, target=target, template_key='direct')
    print (task.config_space)
    # run tuning tasks
    print("Tuning...")
    print (task)
    tune_tasks(task, **tuning_option, n_trial=trials)

def test_bitserial_dense(trials):
    in_dtype = 'uint8'
    pack_dtype='uint8'
    out_dtype = 'int16'
    dorefa = True

    tune_and_evaluate(32, 256, 64, 2, 1, in_dtype, pack_dtype, out_dtype, dorefa, trials)

    verify_bitserial_dense(32, 256, 62, 2, 1, in_dtype, pack_dtype, out_dtype, dorefa)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        trials = int(sys.argv[1])
    else:
        trials = DEFAULT_TRIALS
    test_bitserial_dense(trials)

