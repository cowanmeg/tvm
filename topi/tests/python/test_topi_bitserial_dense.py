"""Example code to do bitserial dense."""
import os
import numpy as np
import tvm
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
from tvm.contrib import rpc, util
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

target = tvm.target.create('llvm -target=x86_64-linux-gnu -mcpu=core-avx2')
device_key = 'x86'
log_file =  os.environ["TVM_ROOT"] + '/tuner/logs/bitserial_dense_x86.log'

def generate_quantized_np(shape, bits, out_dtype):
    np.random.seed(0)
    min_val = 0
    max_val = 1 << bits
    return np.random.randint(min_val, max_val, size=shape).astype(out_dtype)

def verify_bitserial_dense(batch, in_dim, out_dim, data_bits, weight_bits, in_dtype, pack_dtype, out_dtype, dorefa):
    with autotvm.apply_history_best(log_file):
        with tvm.target.create('llvm'):
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
        if dorefa:
            for x in np.nditer(w_, op_flags=['readwrite']):
                x[...] = 1 if x == 1 else -1
        b_np = np.dot(a_np.astype(out_dtype), w_.T.astype(out_dtype))
        return a_np, w_np, b_np
    a_np, w_np, b_np = get_ref_data()

    ctx = tvm.cpu(0)
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func = tvm.build(s, [A, W, B], "llvm")
    func(a, w, b)
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

def test_bitserial_dense(batch, in_dim, out_dim):
    in_dtype = 'uint16'
    pack_dtype='uint8'
    out_dtype = 'int16'
    dorefa = True

    verify_bitserial_dense(batch, in_dim, out_dim, 1, 1, in_dtype, pack_dtype, out_dtype, dorefa)
    verify_bitserial_dense(batch, in_dim, out_dim, 2, 1, in_dtype, pack_dtype, out_dtype, dorefa)

    in_dtype = 'uint16'
    pack_dtype='uint8'
    out_dtype = 'uint16'
    dorefa = False

    verify_bitserial_dense(batch, in_dim, out_dim, 1, 1, in_dtype, pack_dtype, out_dtype, dorefa)
    verify_bitserial_dense(batch, in_dim, out_dim, 2, 1, in_dtype, pack_dtype, out_dtype, dorefa)

if __name__ == "__main__":
    # test_bitserial_dense(1, 1024, 1000)
    # test_bitserial_dense(1, 1024, 10)
    test_bitserial_dense(1, 8, 10)


