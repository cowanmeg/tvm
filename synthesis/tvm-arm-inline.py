import os
import tvm
import numpy as np
from tvm.contrib import util, clang
from tvm import autotvm, rpc
import topi.testing

a_dtype = "uint8"
b_dtype = "uint16"
np.random.seed(0)

def impl():
    assembly = "assembly.c"
    intrin = "intrin.c"
    src = open(os.path.join(os.path.dirname(__file__), assembly)).read()
    return clang.create_llvm(src, options=["-O3", "--target=armv7-none-linux-gnueabihf", "-mcpu=cortex-a53", "-mfpu=neon"])

def intrin():
        m = 8
        l = 16
        
        a = tvm.placeholder((m, l), name='a', dtype=a_dtype)
        k = tvm.reduce_axis((0, l), name='k')
        b = tvm.compute((m,), lambda i: tvm.sum(a[i, k].astype(b_dtype), axis=k), name='b')
        Ab = tvm.decl_buffer(a.shape, a.dtype,
                            name="A",
                            offset_factor=1,
                            strides=[tvm.var("s1"), 1])
        Bb = tvm.decl_buffer(b.shape, b.dtype,
                            name="B",
                            offset_factor=1,
                            strides=[1])

        def intrin_func(ins, outs):
            aa = ins[0]
            bb = outs[0]
            def _body():
                ib = tvm.ir_builder.create()
                ib.emit(tvm.call_extern("int32", "update",
                                    aa.access_ptr("r"),
                                    bb.access_ptr("rw")))

                return ib.get()
            def _reduce_reset():
                ib = tvm.ir_builder.create()
                ib.emit(tvm.call_extern("int32", "reset",
                                    bb.access_ptr("w")))
                return ib.get()
            def _reduce_update():
                return _body()
            return _body(), _reduce_reset(), _reduce_update()
        with tvm.build_config(offset_factor=1):
            return tvm.decl_tensor_intrin(b.op, intrin_func, binds={a: Ab, b: Bb})

def test_arm_inline():
    K = 16
    M = 16
    ashape = (M, K)
    bshape = (M, )

    ktile = 16
    mtile = 8


    A = tvm.placeholder(ashape, dtype=a_dtype, name='A')
    k = tvm.reduce_axis((0, K), name='k')
    A_tiled = tvm.compute((M//mtile, K//ktile, mtile, ktile),
            lambda mo, ko, mi, ki:
            A[mo*mtile + mi, ko*ktile + ki])
    B = tvm.compute(bshape, lambda i:
            tvm.sum(A_tiled[i//mtile, k//ktile, i%mtile, k%ktile].astype(b_dtype), axis=k), name="B")
    s = tvm.create_schedule(B.op)
    print(tvm.lower(s, [A,B], simple_mode=True))
    x = B.op.axis[0]
    z = B.op.reduce_axis[0]
    xo, xi = s[B].split(x, factor=mtile)
    zo, zi = s[B].split(z, factor=ktile)
    s[B].reorder(xo, zo, xi, zi)
    s[B].tensorize(xi, intrin())
    print(tvm.lower(s, [A, B], simple_mode=True))
    s[B].pragma(xo, "import_llvm", impl())
    print(tvm.lower(s, [A, B], simple_mode=True))

    target = tvm.target.arm_cpu("rasp3b")
    with target:
    # BUILD and invoke the kernel.
      f = tvm.build(s, [A, B], target=target)
      f.save(os.path.join(os.getcwd(), 'test.ll'))
      f.save(os.path.join(os.getcwd(), 'test.asm'))
    # Running on standalone device
      # host = '10.77.1.69'
      # port = 9090
      # remote = rpc.connect(host, port)
      # ctx = remote.cpu()

      remote = autotvm.measure.request_remote('rpi3b', 'fleet.cs.washington.edu', 9190, timeout=10000)
      ctx  = remote.context(str(target))
      temp = util.tempdir()
      path = temp.relpath('lib.tar')
      f.export_library(path)
      remote.upload(path)
      f = remote.load_module('lib.tar')
      # launch the kernel.
      a = tvm.nd.array(np.random.randint(0, 15, size=ashape).astype(A.dtype), ctx)
      b = tvm.nd.array(np.random.uniform(size=bshape).astype(B.dtype), ctx)
      f(a, b)

      # Correct
      a_np = a.asnumpy()
      b_np = np.sum(a_np, axis=1).astype(b_dtype)
      #print(b_np)
      #print(a.asnumpy(), b.asnumpy())
      tvm.testing.assert_allclose(b_np, b.asnumpy(), rtol=1e-5)
      
test_arm_inline()
