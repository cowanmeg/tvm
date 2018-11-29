import os
import numpy as np
import tvm
import topi
import topi.testing
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize
from tvm.autotvm.task.nnvm_integration import serialize_args

np.random.seed(0)

TRIALS = 25
target = tvm.target.create('llvm -target=x86_64-linux-gnu -mcpu=core-avx2')
device_key = 'x86'
log_file =  os.environ["TVM_ROOT"] + '/tuner/logs/bitserial_conv2d_x86.log'

tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'early_stopping': 400,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(
            build_func='default'),
        runner=autotvm.LocalRunner(
            number=5,
            timeout=5,
        ),
    ),
}

def generate_quantized_np(shape, bits, out_dtype):
    min_val = 0
    max_val = 1 << bits
    return np.random.randint(min_val, max_val, size=shape).astype(out_dtype)

def verify_bitserial_conv2d_nchw(batch, in_size, in_channel, num_filter, kernel, stride, padding,
    activation_bits, weight_bits, in_dtype, pack_dtype, out_dtype, dorefa):
    in_height = in_width = in_size
    with autotvm.apply_history_best(log_file):
        with tvm.target.create('llvm'):
            A = tvm.placeholder((batch, in_channel, in_height, in_width), dtype=in_dtype, name='A')
            W = tvm.placeholder((num_filter, in_channel, kernel, kernel), dtype=in_dtype, name='W')
            B = topi.nn.bitserial_conv2d_nchw(A, W, stride, padding, activation_bits, weight_bits,
                                              pack_dtype, out_dtype, dorefa)
            s = topi.generic.schedule_bitserial_conv2d_nchw([B])

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)

    @memoize("topi.tests.test_topi_bitseral_conv2d_nchw")
    def get_ref_data():
        a_np = generate_quantized_np(get_const_tuple(a_shape), activation_bits, in_dtype)
        w_np = generate_quantized_np(get_const_tuple(w_shape), weight_bits, in_dtype)
        if dorefa:
            w_ = np.copy(w_np).astype(out_dtype)
            for x in np.nditer(w_, op_flags=['readwrite']):
                x[...] = 1 if x == 1 else -1
            b_np = topi.testing.conv2d_nchw_python(a_np.astype(out_dtype), w_, stride, padding)
        else:
            b_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding)
        return a_np, w_np, b_np
    a_np, w_np, b_np = get_ref_data()

    ctx = tvm.cpu(0)
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func = tvm.build(s, [A, W, B], "llvm")
    func(a, w, b)
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

def verify_bitserial_conv2d_nhwc(batch, in_size, in_channel, num_filter, kernel, stride, padding,
                        activation_bits, weight_bits, in_dtype, pack_dtype, out_dtype, dorefa):
    in_height = in_width = in_size
    with autotvm.apply_history_best(log_file):
        with tvm.target.create('llvm'):
            A = tvm.placeholder((batch, in_height, in_width, in_channel), dtype=in_dtype, name='A')
            W = tvm.placeholder((kernel, kernel, in_channel, num_filter), dtype=in_dtype, name='W')
            B = topi.nn.bitserial_conv2d_nhwc(A, W, stride, padding, activation_bits, weight_bits, pack_dtype, out_dtype, dorefa)
            # s = tvm.create_schedule(B.op)
            s = topi.generic.schedule_bitserial_conv2d_nhwc([B])

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)

    def get_ref_data():
        a_np = generate_quantized_np(get_const_tuple(a_shape), activation_bits, in_dtype)
        w_np = generate_quantized_np(get_const_tuple(w_shape), weight_bits, in_dtype)
        if dorefa:
            w_ = np.copy(w_np).astype(out_dtype)
            for x in np.nditer(w_, op_flags=['readwrite']):
                x[...] = 1 if x == 1 else -1
            b_np = topi.testing.conv2d_nhwc_python(a_np, w_, stride, padding).astype(out_dtype)
        else:
            b_np = topi.testing.conv2d_nhwc_python(a_np, w_np, stride, padding).astype(out_dtype)
        return a_np, w_np, b_np
    a_np, w_np, b_np = get_ref_data()

    ctx = tvm.cpu(0)
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func = tvm.build(s, [A, W, B], 'llvm')

    func(a, w, b)
    print(b)
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

def tune_tasks(tsk,
               measure_option,
               tuner='xgb',
               n_trial=TRIALS,
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
    os.remove(tmp_log_file)

def tune_and_evaluate(batch, in_size, in_channel, num_filter, kernel, stride, padding,
                        activation_bits, weight_bits, in_dtype, pack_dtype, out_dtype, dorefa, layout='nhwc'):
    in_height = in_width = in_size

    if layout == "nchw":
        A = tvm.placeholder((batch, in_channel, in_height, in_width), dtype=in_dtype, name='A')
        W = tvm.placeholder((num_filter, in_channel, kernel, kernel), dtype=in_dtype, name='W')
    else:
        A = tvm.placeholder((batch, in_height, in_width, in_channel), dtype=in_dtype, name='A')
        W = tvm.placeholder((kernel, kernel, in_channel, num_filter), dtype=in_dtype, name='W')
    args = [A, W, stride, padding, activation_bits, weight_bits, pack_dtype, out_dtype, dorefa]
    args = serialize_args(args)
    task = autotvm.task.create("topi_x86_bitserial_conv_" + layout,
                       args=args, target=target, template_key='direct')
    print (task.config_space)
    # run tuning tasks
    print("Tuning...")
    print (task)
    tune_tasks(task, **tuning_option)

def test_bitserial_conv2d(in_size, ic, oc, k, stride, pad):
    in_dtype = 'uint8'
    pack_dtype = 'uint64' # TODO: Not working for pack dtypes of 32 and 64
    out_dtype = 'int16'
    dorefa = True

    # tune_and_evaluate(1, in_size, ic, oc, k, stride, pad, 2, 1, in_dtype, pack_dtype, out_dtype, dorefa, "nhwc")
    # print ("verify")

    verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 2, 1, in_dtype, pack_dtype, out_dtype, dorefa)
    # verify_bitserial_conv2d_nchw(1, in_size, ic, oc, k, stride, pad, 2, 1, True)
    # verify_bitserial_conv2d_nchw(1, in_size, ic, oc, k, stride, pad, 1, 1, False)
    # verify_bitserial_conv2d_nchw(1, in_size, ic, oc, k, stride, pad, 2, 1, False)
    # verify_bitserial_conv2d_nchw(1, in_size, ic, oc, k, stride, pad, 2, 2, False)

    # verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 1, 1, True)
    # verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 2, 1, True)
    # verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 1, 1, False)
    # verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 2, 1, False)
    # verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 2, 2, False)

if __name__ == "__main__":
    # test_bitserial_conv2d(64, 32, 64, 3, 2, 1)
    test_bitserial_conv2d(64, 64, 128, 3, 2, 1)
