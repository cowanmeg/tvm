"""Example code to do qconv2d."""
import os
import numpy as np
import tvm
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
from tvm.contrib import rpc, util


def generate_quantized_np(shape, bits, out_dtype):
    np.random.seed(0)
    min_val = 0
    max_val = 1 << bits
    return np.random.randint(min_val, max_val, size=shape).astype(out_dtype)

def verify_bitserial_dense(batch, in_dim, out_dim, data_bits, weight_bits):
    input_type='int8'
    out_dtype='int32'

    with tvm.target.create('llvm'):
        A = tvm.placeholder((batch, in_dim), dtype=input_type, name='A')
        W = tvm.placeholder((out_dim, in_dim), dtype=input_type, name='W')
        B = topi.nn.bitserial_dense(A, W, data_bits, weight_bits, out_dtype=out_dtype)
        s = topi.generic.schedule_bitserial_dense(B)
        # s = tvm.create_schedule(B.op)
        # print (tvm.lower(s, [A, W, B], simple_mode=True))

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)

    def get_ref_data():
        a_np = generate_quantized_np(get_const_tuple(A.shape), data_bits, input_type)
        w_np = generate_quantized_np(get_const_tuple(W.shape), weight_bits, input_type)
        w_ = np.copy(w_np).astype(out_dtype)
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
    verify_bitserial_dense(batch, in_dim, out_dim, 1, 1)
    verify_bitserial_dense(batch, in_dim, out_dim, 2, 1)

if __name__ == "__main__":
    test_bitserial_dense(1, 1024, 1000)
    test_bitserial_dense(1, 1024, 10)

