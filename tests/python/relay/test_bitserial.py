import tvm
from tvm import relay
from tvm.relay.testing import ctx_list
import numpy as np
import topi.testing

def generate_quantized_np(shape, bits, out_dtype):
    min_val = 0
    max_val = 1 << bits
    return np.random.randint(min_val, max_val, size=shape).astype(out_dtype)

def test_conv2d_run():
    def run_test_bitserial_conv2d(dtype, out_dtype, scale, dshape, kshape,
                                  activation_bits=2, weight_bits=1,
                                  padding=(1, 1, 1, 1), channels=64,
                                  unipolar=True, **attrs):
        x = relay.var("x", shape=dshape, dtype="uint8")
        w = relay.var("w", dtype="uint8")
        y = relay.nn.bitserial_conv2d(x, w,
                            padding=padding,
                            activation_bits=activation_bits,
                            weight_bits=weight_bits,
                            channels=channels,
                            pack_dtype="uint32",
                            out_dtype=out_dtype,
                            unipolar=unipolar,
                            **attrs)
        func = relay.Function([x, w], y)
        data = generate_quantized_np(dshape, activation_bits, dtype)
        kernel = generate_quantized_np(kshape, weight_bits, dtype)
        if unipolar:
            kernel_ = np.copy(kernel).astype(out_dtype)
            for x in np.nditer(kernel_, op_flags=['readwrite']):
                x[...] = 1 if x == 1 else -1
        else:
            kernel_ = kernel
        ref_res = topi.testing.conv2d_nhwc_python(
            data.astype(out_dtype), kernel_.astype(out_dtype), 1, 1)


        for target, ctx in ctx_list():
            intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
            op_res1 = intrp1.evaluate(func)(data, kernel)
            tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5, atol=1e-5)

    # Unipolar = False
    dshape = (1, 224, 224, 64)
    kshape = (3, 3, 64, 64)
    run_test_bitserial_conv2d("uint16", "int16", 1, dshape, kshape, unipolar=False,
                              padding=(1, 1, 1, 1), channels=64, kernel_size=(3, 3))

    # Unipolar = True
    run_test_bitserial_conv2d("uint16", "int16", 1, dshape, kshape, unipolar=True,
                              padding=(1, 1, 1, 1), channels=64, kernel_size=(3, 3))

if __name__ == '__main__':
    test_conv2d_run()
    

