import os
import tvm
import numpy as np
from tvm.contrib import util, clang
from tvm import autotvm, rpc
import topi.testing

a_dtype = "uint8"
c_dtype = "uint16"
np.random.seed(0)

def impl():
    assembly = "assembly.c"
    intrin = "ukernel-intrin.c"
    src = open(os.path.join(os.path.dirname(__file__), intrin)).read()
    return clang.create_llvm(src, options=["-O3", "--target=armv7-none-linux-gnueabihf", "-mcpu=cortex-a53", "-mfpu=neon"])

def intrin():
        m = 8
        l = 16
        
        a = tvm.placeholder((m, l), name='a', dtype=a_dtype)
        b = tvm.placeholder((1, l), name='b', dtype=a_dtype)
        k = tvm.reduce_axis((0, l), name='k')
        c = tvm.compute((m,), lambda i: 
            tvm.sum(tvm.popcount(a[i, k] & b[0, k]).astype(c_dtype), axis=k), name='c')
        Ab = tvm.decl_buffer(a.shape, a.dtype,
                            name="A",
                            offset_factor=1,
                            strides=[tvm.var("s1"), 1])
        Bb = tvm.decl_buffer(b.shape, a.dtype,
                            name="B",
                            offset_factor=1,
                            strides=[tvm.var("s2"), 1])

        Cb = tvm.decl_buffer(c.shape, c.dtype,
                            name="C",
                            offset_factor=1,
                            strides=[1])
        def intrin_func(ins, outs):
            aa = ins[0]
            bb = ins[1]
            cc = outs[0]
            def _body():
                ib = tvm.ir_builder.create()
                ib.emit(tvm.call_extern("int32", "update",
                                    aa.access_ptr("r"),
                                    bb.access_ptr("r"),
                                    cc.access_ptr("rw")))

                return ib.get()
            def _reduce_reset():
                ib = tvm.ir_builder.create()
                ib.emit(tvm.call_extern("int32", "reset",
                                    cc.access_ptr("w")))
                return ib.get()
            def _reduce_update():
                return _body()
            return _body(), _reduce_reset(), _reduce_update()
        with tvm.build_config(offset_factor=1):
            return tvm.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})

def test_arm_inline():
    K = 32
    M = 16
    ashape = (M, K)
    bshape = (1, K)
    cshape = (M, )

    ktile = 16
    mtile = 8


    A = tvm.placeholder(ashape, dtype=a_dtype, name='A')
    B = tvm.placeholder(bshape, dtype=a_dtype, name='B')
    k = tvm.reduce_axis((0, K), name='k')
    A_tiled = tvm.compute((M//mtile, K//ktile, mtile, ktile),
            lambda mo, ko, mi, ki:
            A[mo*mtile + mi, ko*ktile + ki])
    C = tvm.compute(cshape, lambda i:
            tvm.sum(tvm.popcount(A_tiled[i//mtile, k//ktile, i%mtile, k%ktile] &  B[0, k]).astype(c_dtype), axis=k), name="C")
    s = tvm.create_schedule(C.op)
    print(tvm.lower(s, [A,B,C], simple_mode=True))
    x = C.op.axis[0]
    z = C.op.reduce_axis[0]
    xo, xi = s[C].split(x, factor=mtile)
    zo, zi = s[C].split(z, factor=ktile)
    s[C].reorder(xo, zo, xi, zi)
    s[C].tensorize(xi, intrin())
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    s[C].pragma(xo, "import_llvm", impl())
    print(tvm.lower(s, [A, B, C], simple_mode=True))

    target = tvm.target.arm_cpu("rasp3b")
    with target:
    # BUILD and invoke the kernel.
      f = tvm.build(s, [A, B, C], target=target)
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
      b = tvm.nd.array(np.random.randint(0, 25, size=bshape).astype(B.dtype), ctx)
      c = tvm.nd.array(np.random.uniform(size=cshape).astype(C.dtype), ctx)
      f(a, b, c)

      # Correct
      a_np = a.asnumpy()
      b_np = b.asnumpy()
      c1_np = np.bitwise_and(a_np, b_np)
      mask = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]
      for i in range(0, M):
          for j in range(0, K):
            t = c1_np[i, j]
            c1_np[i, j] = 0
            for l in range(0, 8):
                c1_np[i, j] += np.bitwise_and(t, mask[l]) >> l
      c_np = np.sum(c1_np, axis=1).astype(c_dtype)
      print(c_np)
      print(c)
      #print(a.asnumpy(), b.asnumpy())
      tvm.testing.assert_allclose(c_np, c.asnumpy(), rtol=1e-5)
      
test_arm_inline()
