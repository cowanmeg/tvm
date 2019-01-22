"""Bitserial conv2d schedule on raspberdh pi"""
from __future__ import absolute_import as _abs
import numpy as np
from collections import namedtuple
import os
import tvm
import topi
import topi.testing
from tvm import autotvm, rpc
from topi import tag
from topi.nn.pad import pad
from topi.nn.bitserial_conv2d import bitpack, bitserial_conv2d_nhwc
from topi.nn.util import get_pad_tuple
from topi.util import get_const_int, get_const_tuple
from tvm.autotvm.task.nnvm_integration import deserialize_args
# from topi.arm_cpu.bitserial_conv2d import _intrin_popcount, fused_conv
from tvm.contrib import util

input_type='uint16'
out_dtype='int16'
pack_dtype='uint8'

def get_padding(padding, kernel_h, kernel_w):
    if padding is 'VALID':
        pad_h = 0 
        pad_w = 0 
    elif padding is 'SAME':
        pad_h = kernel_h - 1
        pad_w = kernel_w - 1
    else: # Padding given as tuple
        return padding
    pad_top = int(np.ceil(float(pad_h) / 2))
    pad_bottom = pad_h - pad_top
    pad_left = int(np.ceil(float(pad_w) / 2))
    pad_right = pad_w - pad_left
    return (pad_top, pad_left, pad_bottom, pad_right)

def generate_quantized_np(shape, bits, out_dtype):
    min_val = 0
    max_val = 1 << bits
    return np.random.randint(min_val, max_val, size=shape).astype(out_dtype)

def _kernel_vec_spatial_pack_nhwc(kernel, kernel_bits, VC, use_bitpack=True):
    if use_bitpack:
        kernel_q = bitpack(kernel, kernel_bits, pack_axis=2, bit_axis=2, pack_type='uint8')
    else:
        kernel_q = kernel
    KH, KW, KB, CI, CO = kernel_q.shape
    kvshape = (CO//VC, KH, KW, KB, VC, CI)
    return tvm.compute(kvshape, lambda co, dh, dw, b, vc, ci: \
        kernel_q[dh][dw][b][ci][co*VC+vc], name='kernel_vec')

# For comparison against
def simple_spatial_pack_nhwc_packed(data, kernel, r, clip_min, clip_max, rshift,
                      stride, padding, activation_bits, weight_bits, 
                      pack_dtype, out_dtype, dorefa, pool_size, pool_stride, pool_pad):
    """ Compute convolution with pack on spatial axes. """

    assert isinstance(stride, int) or len(stride) == 2
    Input_q = bitpack(data, activation_bits, pack_axis=3, bit_axis=1, pack_type=pack_dtype)
    Filter_q = bitpack(kernel, weight_bits, pack_axis=2, bit_axis=0, pack_type=pack_dtype)
    weight_bits, kernel_h, kernel_w, in_channel_q, out_channel = get_const_tuple(Filter_q.shape)
    batch, _, in_height, in_width, in_channel_q = get_const_tuple(Input_q.shape)

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    pad_top, pad_left, pad_down, pad_right = padding
    # compute the output shape
    out_height = (in_height - kernel_h + pad_top + pad_down) // stride_h + 1
    out_width = (in_width - kernel_w + pad_left + pad_right) // stride_w + 1
    pad_before = [0, 0, pad_top, pad_left, 0]
    pad_after = [0, 0, pad_down, pad_right, 0]
    PadInput_q = pad(Input_q, pad_before, pad_after, name="PaddedInput")

    ci = tvm.reduce_axis((0, in_channel_q), name='ci')
    dh = tvm.reduce_axis((0, kernel_h), name='dh')
    dw = tvm.reduce_axis((0, kernel_w), name='dw')
    ib = tvm.reduce_axis((0, activation_bits), name='ib')
    kb = tvm.reduce_axis((0, weight_bits), name='kb')

    def _conv(nn, yy, xx, ff):
        b1b2 = (ib+kb).astype(out_dtype)
        return tvm.sum(
            ((tvm.popcount(
                PadInput_q[nn, ib, yy * stride_h + dh, xx * stride_w + dw, ci] &
                Filter_q[kb, dh, dw, ci, ff]).astype(out_dtype) - 
             tvm.popcount(
                PadInput_q[nn, ib, yy * stride_h + dh, xx * stride_w + dw, ci] &
                ~Filter_q[kb, dh, dw, ci, ff]).astype(out_dtype))<< b1b2),
                       axis=[ci, dh, dw, kb, ib])
    conv = tvm.compute((batch, out_height, out_width, out_channel), _conv,
        name="Conv2dOutput", tag="bitserial_conv2d_nhwc")

    
    rounded_conv = conv + r
    rounded_conv = topi.clip_channelwise(rounded_conv, clip_min, clip_max, axis=3)
    rounded_conv = topi.right_shift_channelwise(rounded_conv, rshift, axis=3)
    pool = topi.nn.pool(rounded_conv, kernel=pool_size, stride=pool_stride, padding=pool_pad, 
        pool_type='max', layout='NHWC')
    packed_conv = bitpack(pool, activation_bits, pack_axis=3, bit_axis=3, pack_type=pack_dtype)

    return conv, packed_conv

def spatial_pack_nhwc_packed(data, kernel, stride, padding, activation_bits, weight_bits, 
                      pack_dtype, out_dtype, dorefa, pool_kernel, pool_stride, pool_pad):
    """ Compute convolution with pack on spatial axes. """

    # assert isinstance(stride, int) or len(stride) == 2
    # Input_q = bitpack(data, activation_bits, pack_axis=3, bit_axis=3, pack_type=pack_dtype)
    # Filter_q = bitpack(kernel, weight_bits, pack_axis=2, bit_axis=0, pack_type=pack_dtype)
    # weight_bits, kernel_h, kernel_w, in_channel_q, out_channel = get_const_tuple(Filter_q.shape)
    # batch, in_height, in_width, _, in_channel_q = get_const_tuple(Input_q.shape)
    # # move input channels to innermost of Filter and split output_channels so it's a multiple of 8
    # VC = 8 
    # Filter_vec = tvm.compute((kernel_h, kernel_w, weight_bits, out_channel, in_channel_q), lambda kh, kw, b, oc, ic: \
    #     Filter_q[b][kh][kw][ic][oc], name='Filter')

    # if isinstance(stride, int):
    #     stride_h = stride_w = stride
    # else:
    #     stride_h, stride_w = stride
    # pad_top, pad_left, pad_down, pad_right = padding
    # # compute the output shape
    # out_height = (in_height - kernel_h + pad_top + pad_down) // stride_h + 1
    # out_width = (in_width - kernel_w + pad_left + pad_right) // stride_w + 1
    # pad_before = [0, pad_top, pad_left, 0, 0]
    # pad_after = [0, pad_down, pad_right, 0, 0]
    # PadInput_q = pad(Input_q, pad_before, pad_after, name="Input")
    # print (Input_q, PadInput_q)

    # ci = tvm.reduce_axis((0, in_channel_q), name='ci')
    # dh = tvm.reduce_axis((0, kernel_h), name='dh')
    # dw = tvm.reduce_axis((0, kernel_w), name='dw')
    # ib = tvm.reduce_axis((0, activation_bits), name='ib')
    # kb = tvm.reduce_axis((0, weight_bits), name='kb')

    # Fused conv does bitserial conv + shift norm + pooling + packing
    # def fused_conv(nn, yy, xx, ff):
    #     b1b2 = (ib+kb).astype(out_dtype)
    #     return tvm.sum(
    #                 ((tvm.popcount(
    #                     Filter_vec[dh, dw, kb, ff, ci].astype(out_dtype) &
    #                     PadInput_q[nn, yy*stride_h+dh, xx*stride_w+dw, ib, ci].astype(out_dtype)) - 
    #                 tvm.popcount(  
    #                     ~Filter_vec[dh, dw, kb, ff, ci].astype(out_dtype) &
    #                     PadInput_q[nn, yy*stride_h+dh, xx*stride_w+dw, ib, ci].astype(out_dtype)))
    #             << b1b2), axis=[dh, dw, kb, ib, ci])

    # conv = tvm.compute((batch, out_height, out_width, out_channel), fused_conv, name="Conv2dOutput", tag="bitserial_conv2d_nhwc")
    conv = topi.nn.bitserial_conv2d_nhwc(data, kernel, stride, padding, activation_bits, weight_bits, 
        pack_dtype=pack_dtype, out_dtype=out_dtype, dorefa=dorefa)
    # TODO: Change to the shiftnorm style (addition + shift) - Something going on here
    # quantized_conv = tvm.compute(conv.shape, lambda *idx: tvm.select(conv(*idx) < 255, conv(*idx).astype('int16'), tvm.const(255, 'int16')), name='quantized')
    pooled = topi.nn.pool(conv, kernel=pool_kernel, stride=pool_stride, padding=pool_pad, pool_type='max', layout='NHWC')
    
    # Some way to make this faster?
    masks = np.array([0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80])
    def _bitpack(nn, bb, yy, xx, ff):
        packed_data = tvm.const(0, pack_dtype)
        data_width = 8
        mask = tvm.select(bb==0, 0x1, 0x2)
        for k in range(data_width):
            f = ff * data_width + k
            element = pooled[nn, yy, xx, f]
            extracted_bit = ((element & mask) >> bb).astype(pack_dtype)
            packed_data = (packed_data | extracted_bit)
            if k < data_width - 1:
                packed_data = packed_data << 1
        return packed_data

    n, h, w, c = get_const_tuple(pooled.shape)
    packed_conv = tvm.compute((n, activation_bits, h, w, c//8), _bitpack, name='BitpackedConv')
    # padded_conv = pad(packed_conv, [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], name="Out")
    return conv, pooled, packed_conv
 
# ARM specific schedule that using custom microkernel
def schedule_packed(data, kernel, conv, pooled, packed_conv, padded_conv, dorefa):
    s = tvm.create_schedule(padded_conv.op)
    # N, H, W, CI = get_const_tuple(data.shape)
    # KH, KW, _, CO = get_const_tuple(kernel.shape)
    # CI_packed = CI // 8

    # n, oh, ow, co = s[conv].op.axis
    # kh, kw, kb, ib, ci = s[conv].op.reduce_axis
    # s[conv].reorder(n, oh, ow, kh, kw, kb, ib, co, ci)

    # pc = _intrin_popcount(CO, CI_packed, 1, 2, dorefa)
    # s[conv].tensorize(kb, pc)  

    # # n, oh, ow, co = s[quantized_conv].op.axis
    # # s[conv].compute_at(s[quantized_conv], co)

    # n, oh, ow, co = s[pooled].op.axis
    # s[conv].compute_at(s[pooled], ow)

    # b, n, oh, ow, co_packed = s[packed_conv].op.axis
    # s[pooled].compute_at(s[packed_conv], ow)
    
    # b, n, oh, ow, co_packed = s[padded_conv].op.axis
    # s[packed_conv].compute_at(s[padded_conv], b)
    
    # s[padded_conv].parallel(b)
    # s = s.normalize()

    conv_out = conv.op.input_tensors[0]

    n, oh, ow, co, vh, vw, vc = s[conv_out].op.axis
    kh, kw, kb, ib, ci = s[conv_out].op.reduce_axis

    ci_o, ci_i = cfg['tile_ci'].apply(s, conv_out, ci)
    re_axes = cfg["reorder_0"].apply(s, conv_out, [n, oh, ow, co, vh, vw, kh, kw, ci_o, kb, ib, vc, ci_i])
    
    kfactor = cfg['tile_ci'].size[1]
    pc = _intrin_popcount(VC, kfactor, KB, IB, dorefa)
    s[conv_out].tensorize(kb, pc)  

    n, h, w, co = s[conv].op.axis
    co, vc = s[conv].split(co, VC)
    oh, ow, vh, vw = s[conv].tile(h, w, VH, VW)
    s[conv].reorder(n, oh, ow, co, vc, vh, vw)
    s[conv].vectorize(vw)

    oho, iho = cfg["tile_oh"].apply(s, last, oh)  # reuse parameter
    s[conv_out].compute_at(s[conv], ow)
    if cfg["tile_oh"].size[1] == 1:
        oaxis = oho
        paxis = oho
    else:
        oaxis = oho
        paxis = iho

    s[conv].parallel(paxis)
    s = s.normalize()

    return s


# Testing:
def solution(batch, in_size, in_channel, num_filter, kernel, stride, padding, 
                        activation_bits, weight_bits, dorefa, pool_size, pool_stride, pool_pad):
    in_height = in_width = in_size
    pad = get_padding(padding, kernel, kernel)
    with tvm.target.create('llvm'):
        A = tvm.placeholder((batch, in_height, in_width, in_channel), dtype=input_type, name='A')
        W = tvm.placeholder((kernel, kernel, in_channel, num_filter), dtype=input_type, name='W')
        Round = tvm.placeholder((in_channel,), dtype=out_dtype, name='Round')
        ClipMin = tvm.placeholder((in_channel,), dtype=out_dtype, name='ClipMin')
        ClipMax = tvm.placeholder((in_channel,), dtype=out_dtype, name='ClipMax')
        RShift = tvm.placeholder((in_channel,), dtype=out_dtype, name='RShift')
        B0, B = simple_spatial_pack_nhwc_packed(A, W, Round, ClipMin, ClipMax, RShift,
                                    stride, pad, activation_bits, weight_bits, 
                                    pack_dtype, out_dtype, dorefa, 
                                    pool_size, pool_stride, pool_pad)
        s = tvm.create_schedule([B0.op, B.op])

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)

    def get_ref_data():
        a_np = generate_quantized_np(get_const_tuple(a_shape), activation_bits, input_type)
        w_np = generate_quantized_np(get_const_tuple(w_shape), weight_bits, input_type)
        r_np = np.random.randint(0, 4, size=(num_filter,)).astype(out_dtype)
        clipmin_np = np.zeros(shape=(num_filter,)).astype(out_dtype)
        clipmax_np = np.random.randint(250, 255, size=(num_filter,)).astype(out_dtype)
        shift_np = np.random.randint(0, 4, size=(num_filter,)).astype(out_dtype)
        if dorefa:
            w_ = np.copy(w_np).astype(out_dtype)
            for x in np.nditer(w_, op_flags=['readwrite']):
                x[...] = 1 if x == 1 else -1
            b_np = topi.testing.conv2d_nhwc_python(a_np, w_, stride, padding).astype(out_dtype)
        else:
            b_np = topi.testing.conv2d_nhwc_python(a_np, w_np, stride, padding).astype(out_dtype)
        return a_np, w_np, r_np, clipmin_np, clipmax_np, shift_np, b_np
    a_np, w_np, r_np, clipmin_np, clipmax_np, shift_np, b_np = get_ref_data()

    ctx = tvm.cpu(0)
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    r = tvm.nd.array(r_np, ctx)
    clipmin = tvm.nd.array(clipmin_np, ctx)
    clipmax = tvm.nd.array(clipmax_np, ctx)
    shift = tvm.nd.array(shift_np, ctx)
    b0 = tvm.nd.array(np.zeros(get_const_tuple(B0.shape), dtype=B0.dtype), ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func = tvm.build(s, [A, W, Round, ClipMin, ClipMax, RShift, B0, B], 'llvm')

    func(a, w, r, clipmin, clipmax, shift, b0, b)
    tvm.testing.assert_allclose(b0.asnumpy(), b_np, rtol=1e-5)
    return a_np, w_np, r_np, clipmin_np, clipmax_np, shift_np, b.asnumpy()

def verify_bitserial_conv2d_nhwc(batch, in_size, in_channel, num_filter, kernel, stride, padding, 
                        activation_bits, weight_bits, dorefa, pooling_kernel, pooling_stride, pooling_pad, 
                        pack_outputs=True):
    in_height = in_width = in_size
    
    pad = get_padding(padding, kernel, kernel)
    with tvm.target.arm_cpu("rasp3b"):
        A = tvm.placeholder((batch, in_height, in_width, in_channel), dtype=input_type, name='A')
        W = tvm.placeholder((kernel, kernel, in_channel, num_filter), dtype=input_type, name='W')
        Round = tvm.placeholder((in_channel,), dtype=out_dtype, name='Round')
        ClipMin = tvm.placeholder((in_channel,), dtype=out_dtype, name='ClipMin')
        ClipMax = tvm.placeholder((in_channel,), dtype=out_dtype, name='ClipMax')
        RShift = tvm.placeholder((in_channel,), dtype=out_dtype, name='RShift')

        B = topi.nn.bitserial_conv2d_nhwc(A, W, Round, ClipMin, ClipMax, RShift,
                                    stride, pad, activation_bits, weight_bits, 
                                    pack_dtype, out_dtype, dorefa, 
                                    pool_size=pool_size, pool_pad=pool_pad, pool_stride=pool_stride,
                                    pack_inputs=True, pack_outputs=pack_outputs)
        s = topi.generic.schedule_bitserial_conv2d_nhwc([B])

        # print(tvm.lower(s, [A, W, Round, ClipMin, ClipMax, RShift, B], simple_mode=True))

        a_np, w_np, r_np, clipmin_np, clipmax_np, shift_np, b_np = solution(batch, in_size, in_channel, num_filter, kernel, stride, 
        padding, activation_bits, weight_bits, dorefa, pooling_kernel, pooling_stride, pooling_pad)

    
    target = 'llvm -target=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon'
    host = '10.77.1.69'
    port = 9090
    remote = rpc.connect(host, port)
    ctx = remote.cpu(0)  

    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    r = tvm.nd.array(r_np, ctx)
    clipmin = tvm.nd.array(clipmin_np, ctx)
    clipmax = tvm.nd.array(clipmax_np, ctx)
    shift = tvm.nd.array(shift_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func = tvm.build(s, [A, W, Round, ClipMin, ClipMax, RShift, B], target)
    # func.save(os.path.join(os.getcwd(), 'conv.ll'))
    # func.save(os.path.join(os.getcwd(), 'conv.asm'))

    # Upload to pi
    temp = util.tempdir()
    path = temp.relpath('conv_nhwc.o')
    func.save(path)
    remote.upload(path)
    func = remote.load_module('conv_nhwc.o')

    func(a, w, r, clipmin, clipmax, shift, b)

    # if pack_outputs:
    #     print ("Differences", np.unique(b.asnumpy() - b_np))
    #     tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-3)

    evaluator = func.time_evaluator(func.entry_name, ctx, number=2)
    print('Time: %f ms' % (evaluator(a, w, r, clipmin, clipmax, shift, b).mean * 1.0e3))

    return b


if __name__ == "__main__":
    # test_bitserial_conv2d(56, 64, 64, 3, 1, (1, 1, 1, 1))
    # test_bitserial_conv2d(56, 64, 64, 3, 1, 'SAME')
    in_size = 14
    ic = 512
    oc = 512
    k = 3
    stride = 1
    pool_size = [2, 2]
    pool_stride = [2, 2]
    pool_pad = (0, 0, 0, 0)
    verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, 'SAME', 2, 1, True, pool_size, 
        pool_stride, pool_pad, False)