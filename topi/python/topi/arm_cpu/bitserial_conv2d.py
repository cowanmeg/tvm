# pylint: disable=invalid-name,unused-variable,invalid-name
"""Bitserial conv2d schedule on raspberry pi"""
from __future__ import absolute_import as _abs
from collections import namedtuple
import tvm
import topi
from tvm import autotvm
from .. import tag
from ..nn.pad import pad
from ..nn.bitserial_conv2d import bitpack, bitserial_conv2d_nhwc
from ..nn.util import get_pad_tuple
from ..util import get_const_int, get_const_tuple
from .. import generic
from tvm.autotvm.task.topi_integration import deserialize_args


@autotvm.task.register("topi_nn_bitserial_conv2d_nhwc")
def _topi_bitserial_conv2d(*args, **kwargs):
    args = deserialize_args(args)
    C = topi.nn.bitserial_conv2d_nhwc(*args, **kwargs)
    s = generic.nn.schedule_bitserial_conv2d_nhwc([C])
    data, kernel, bias, clip_min, clip_max, rshift = args[:6]
    # return s, [data, kernel, clip_min, clip_max, rshift, C]
    return s, [data, kernel, C]

# @_get_schedule.register("arm_cpu")
def _get_schedule_bitserial_conv2d(wkl, layout):
    if wkl not in _WORKLOADS:
        raise ValueError("no schedule for such workload: {}".format(wkl))
    idx = _WORKLOADS.index(wkl)
    if layout == "NCHW":
        sch = _QUANTIZED_SCHEDULES_NCHW[idx]
    elif layout == "NHWC":
        sch = _QUANTIZED_SCHEDULES_NHWC[idx]
    return sch

def _kernel_vec_spatial_pack_nhwc(kernel, kernel_bits, VC, use_bitpack=True):
    if use_bitpack:
        kernel_q = bitpack(kernel, kernel_bits, pack_axis=2, bit_axis=2, pack_type='uint8')
    else:
        kernel_q = kernel
    KH, KW, KB, CI, CO = kernel_q.shape
    kvshape = (CO//VC, KH, KW, KB, VC, CI)
    return tvm.compute(kvshape, lambda co, dh, dw, b, vc, ci: \
        kernel_q[dh][dw][b][ci][co*VC+vc], name='kernel_vec')

@autotvm.register_topi_compute(bitserial_conv2d_nhwc, 'arm_cpu', 'direct')
def spatial_pack_nhwc(cfg, data, kernel,
                      stride, padding, activation_bits, weight_bits, 
                      pack_dtype, out_dtype, dorefa):
    """ Compute convolution with pack on spatial axes. """
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    assert pack_dtype == 'uint8', "only support packing into 8 bits"

    N, H, W, CI = get_const_tuple(data.shape)
    if len(kernel.shape) == 4:
        KH, KW, _, CO = get_const_tuple(kernel.shape)
        CI_packed = CI // 8
    else:
        KH, KW, KB, CI_packed, CO = get_const_tuple(kernel.shape)

    # HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)
    TPAD, LPAD, DPAD, RPAD = padding

    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    HCAT, WCAT = KH-1, KW-1

    PAD_H = H + (TPAD + DPAD)
    PAD_W = W + (LPAD + RPAD)
    OH = (PAD_H - KH) // HSTR + 1
    OW = (PAD_W - KW) // WSTR + 1
    oshape = (1, OH, OW, CO)

    # Pad input channels of weights and data when it is not a multiple of 8
    if CI_packed % 8 != 0:
        CI_PAD = CI_packed % 8
        CI_packed += CI_PAD
    else:
        CI_PAD = 0

    # ==================== define configuration space ====================
    n, oh, ow, co = cfg.axis(N), cfg.axis(OH), cfg.axis(OW), cfg.axis(CO)
    ci, kh, kw = cfg.reduce_axis(CI_packed), cfg.reduce_axis(KH), cfg.reduce_axis(KW)
    ib, kb = cfg.reduce_axis(activation_bits), cfg.reduce_axis(weight_bits)

    co, vc = cfg.define_split('tile_co', co, policy='all', num_outputs=2,
                       filter=lambda x: x.size[-1] == 8)
    oh, vh = cfg.define_split('tile_oh', oh, policy='all', num_outputs=2,
                       filter=lambda x: x.size[-1] >= 2)
    ow, vw = cfg.define_split('tile_ow', ow, policy='all', num_outputs=2,
                       filter=lambda x: x.size[-1] >= 2)
    # cfg.define_annotate('ann_reduce', [ib, kb, kh, kw], policy='try_unroll')
    ci_o, ci_i = cfg.define_split("tile_ci", ci, num_outputs=2, 
                                 filter=lambda x: x.size[-1] == 8 or x.size[-1] == 16)
    re_axes = cfg.define_reorder("reorder_0",
                          [n, oh, ow, co, vh, vw, kh, kw, ci_o, kb, ib, vc, ci_i],
                          policy='candidate', 
                          candidate=[
                          [n, oh, ow, co, vh, vw, kh, kw, ci_o, kb, ib, vc, ci_i],
                          [n, oh, ow, co, vh, vw, kw, kh, ci_o, kb, ib, vc, ci_i],
                          ])
    cfg.add_flop(2 * N * OH * OW * CO * CI * 8 * KH * KW)

    if cfg.is_fallback:
        assert True, "Error: Fall back not implemented yet"
    # ====================

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    data_q = bitpack(data, activation_bits, pack_axis=3, bit_axis=3, pack_type='uint8')

    kernel_vec = _kernel_vec_spatial_pack_nhwc(kernel, weight_bits, VC, len(kernel.shape) == 4)
    if kernel_vec.shape[-1] % 8 != 0 and CI_PAD != 0:
        kernel_vec = pad(kernel_vec, [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, CI_PAD])
    
    N, H, W, IB, CI = data_q.shape
    OCO, KH, KW, KB, VC, CI = kernel_vec.shape

    dvshape = (N, PAD_H//(VH*HSTR), PAD_W//(VW*WSTR), VH*HSTR+HCAT, VW*WSTR+WCAT, IB, CI)
    ovshape = (1, OH // VH, OW // VW, CO // VC, VH, VW, VC)

    if (TPAD != 0 and RPAD != 0):
        data_pad = pad(data_q, (0, TPAD, LPAD, 0, 0), (0, DPAD, RPAD, 0, CI_PAD), name="data_pad")
    elif CI_PAD != 0:
        data_pad = pad(data_q, (0, 0, 0, 0, 0), (0, 0, 0, 0, CI_PAD), name="data_pad")
    else:
        data_pad = data_q

    data_vec = tvm.compute(dvshape, lambda n, h, w, vh, vw, b, ci: \
        data_pad[n][h*VH*HSTR+vh][w*VW*WSTR+vw][b][ci], name='data_vec')
    ci = tvm.reduce_axis((0, CI), name='ci')
    dh = tvm.reduce_axis((0, KH), name='dh')
    dw = tvm.reduce_axis((0, KW), name='dw')
    ib = tvm.reduce_axis((0, IB), name='ib')
    kb = tvm.reduce_axis((0, KB), name='kb')

    def _conv(n, h, w, co, vh, vw, vc):
        return tvm.sum((tvm.popcount(
            kernel_vec[co, dh, dw, kb, vc, ci].astype('uint16') &
            data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ib, ci].astype('uint16'))
                        << (kb + ib).astype('uint16')), axis=[dh, dw, kb, ib, ci])
    def _dorefa_conv(n, h, w, co, vh, vw, vc):
        return tvm.sum((tvm.popcount(
            kernel_vec[co, dh, dw, kb, vc, ci].astype('int16') &
            data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ib, ci].astype('int16')) -
            tvm.popcount(~kernel_vec[co, dh, dw, kb, vc, ci].astype('int16') &
            data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ib, ci]).astype('int16'))
                        << (kb + ib).astype('int16'), axis=[dh, dw, kb, ib, ci])
    if dorefa:
        conv_vec = tvm.compute(ovshape, _dorefa_conv, name='conv_vec', tag='dorefa')
    else:
        conv_vec = tvm.compute(ovshape, _conv, name='conv_vec')

    conv =  tvm.compute(oshape, lambda n, h, w, co:
                       conv_vec[n][h//VH][w//VW][co//VC][h%VH][w%VW][co%VC].astype(out_dtype),
                       name='conv', tag='spatial_bitserial_conv_nhwc')

    return conv

# @autotvm.register_topi_compute(bitserial_conv2d_nhwc, 'arm_cpu', 'direct')
def packed(cfg, data, kernel, bias, clip_min, clip_max, rshift,
                      stride, padding, activation_bits, weight_bits, 
                      pack_dtype, out_dtype, dorefa, 
                      pool_size, pool_stride, pool_pad,
                      pack_inputs=True, pack_outputs=False):
    """ Compute convolution with pack on spatial axes. """
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    assert pack_dtype == 'uint8', "only support packing into 8 bits"

    if pack_inputs:
        N, H, W, CI = get_const_tuple(data.shape)
    else:
        N, H, W, _, CI_packed = get_const_tuple(data.shape)
        CI = CI_packed * 8
    if len(kernel.shape) == 4:
        KH, KW, _, CO = get_const_tuple(kernel.shape)
        CI_packed = CI // 8
    else:
        KH, KW, KB, CI_packed, CO = get_const_tuple(kernel.shape)

    # HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)
    TPAD, LPAD, DPAD, RPAD = padding

    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    HCAT, WCAT = KH-1, KW-1

    PAD_H = H + (TPAD + DPAD)
    PAD_W = W + (LPAD + RPAD)
    OH = (PAD_H - KH) // HSTR + 1
    OW = (PAD_W - KW) // WSTR + 1
    oshape = (1, OH, OW, CO)

    # Pad input channels of weights and data when it is not a multiple of 8
    if CI_packed % 8 != 0:
        CI_PAD = CI_packed % 8
        CI_packed += CI_PAD
    else:
        CI_PAD = 0

    # ==================== define configuration space ====================
    n, oh, ow, co = cfg.axis(N), cfg.axis(OH), cfg.axis(OW), cfg.axis(CO)
    ci, kh, kw = cfg.reduce_axis(CI_packed), cfg.reduce_axis(KH), cfg.reduce_axis(KW)
    ib, kb = cfg.reduce_axis(activation_bits), cfg.reduce_axis(weight_bits)

    co, vc = cfg.define_split('tile_co', co, policy='all', num_outputs=2,
                       filter=lambda x: x.size[-1] == 8)
    oh, vh = cfg.define_split('tile_oh', oh, policy='all', num_outputs=2,
                       filter=lambda x: x.size[-1] >= 2)
    ow, vw = cfg.define_split('tile_ow', ow, policy='all', num_outputs=2,
                       filter=lambda x: x.size[-1] >= 2)
    # cfg.define_annotate('ann_reduce', [ib, kb, kh, kw], policy='try_unroll')
    ci_o, ci_i = cfg.define_split("tile_ci", ci, num_outputs=2, 
                                 filter=lambda x: x.size[-1] == 8 or x.size[-1] == 16)
    re_axes = cfg.define_reorder("reorder_0",
                          [n, oh, ow, co, vh, vw, kh, kw, ci_o, kb, ib, vc, ci_i],
                          policy='candidate', 
                          candidate=[
                          [n, oh, ow, co, vh, vw, kh, kw, ci_o, kb, ib, vc, ci_i],
                          [n, oh, ow, co, vh, vw, kw, kh, ci_o, kb, ib, vc, ci_i],
                          ])
    cfg.add_flop(2 * N * OH * OW * CO * CI * 8 * KH * KW)

    if cfg.is_fallback:
        assert True, "Error: Fall back not implemented yet"
    # ====================

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    if pack_inputs:
        data_q = bitpack(data, activation_bits, pack_axis=3, bit_axis=3, pack_type='uint8')
    else:
        data_q = data
    kernel_vec = _kernel_vec_spatial_pack_nhwc(kernel, weight_bits, VC, len(kernel.shape) == 4)
    if kernel_vec.shape[-1] % 8 != 0 and CI_PAD != 0:
        kernel_vec = pad(kernel_vec, [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, CI_PAD])
    
    N, H, W, IB, CI = data_q.shape
    OCO, KH, KW, KB, VC, CI = kernel_vec.shape

    dvshape = (N, PAD_H//(VH*HSTR), PAD_W//(VW*WSTR), VH*HSTR+HCAT, VW*WSTR+WCAT, IB, CI)
    ovshape = (1, OH // VH, OW // VW, CO // VC, VH, VW, VC)

    if (TPAD != 0 and RPAD != 0):
        data_pad = pad(data_q, (0, TPAD, LPAD, 0, 0), (0, DPAD, RPAD, 0, CI_PAD), name="data_pad")
    elif CI_PAD != 0:
        data_pad = pad(data_q, (0, 0, 0, 0, 0), (0, 0, 0, 0, CI_PAD), name="data_pad")
    else:
        data_pad = data_q


    data_vec = tvm.compute(dvshape, lambda n, h, w, vh, vw, b, ci: \
        data_pad[n][h*VH*HSTR+vh][w*VW*WSTR+vw][b][ci], name='data_vec')
    ci = tvm.reduce_axis((0, CI), name='ci')
    dh = tvm.reduce_axis((0, KH), name='dh')
    dw = tvm.reduce_axis((0, KW), name='dw')
    ib = tvm.reduce_axis((0, IB), name='ib')
    kb = tvm.reduce_axis((0, KB), name='kb')

    def _conv(n, h, w, co, vh, vw, vc):
        return tvm.sum((tvm.popcount(
            kernel_vec[co, dh, dw, kb, vc, ci].astype('uint16') &
            data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ib, ci].astype('uint16'))
                        << (kb + ib).astype('uint16')), axis=[dh, dw, kb, ib, ci])
    def _dorefa_conv(n, h, w, co, vh, vw, vc):
        return tvm.sum((tvm.popcount(
            kernel_vec[co, dh, dw, kb, vc, ci].astype('int16') &
            data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ib, ci].astype('int16')) -
            tvm.popcount(~kernel_vec[co, dh, dw, kb, vc, ci].astype('int16') &
            data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ib, ci]).astype('int16'))
                        << (kb + ib).astype('int16'), axis=[dh, dw, kb, ib, ci])
    if dorefa:
        conv_vec = tvm.compute(ovshape, _dorefa_conv, name='conv_vec', tag='dorefa')
    else:
        conv_vec = tvm.compute(ovshape, _conv, name='conv_vec')


    conv =  tvm.compute(oshape, lambda n, h, w, co:
                ((tvm.max(tvm.min(conv_vec[n][h//VH][w//VW][co//VC][h%VH][w%VW][co%VC].astype(out_dtype), 
                    clip_max(co)), tvm.const(0, out_dtype) + bias[co])) >> rshift[co]),

                       name='conv', tag='spatial_bitserial_conv_nhwc')

    if not pack_outputs and pool_size is None:
        return conv

    # conv_bias = conv + bias
    # conv_clip = topi.clip_channelwise(conv, clip_min, clip_max, axis=3)
    # conv_shift = topi.right_shift_channelwise(conv_clip, rshift, axis=3)

    if pool_size is None:
        pooled = conv
    else:
        pooled = topi.nn.pool(conv, kernel=pool_size,
            stride=pool_stride, padding=pool_pad, pool_type='max', layout='NHWC')

    # For some reason this is much faster?
    import numpy as np
    masks = np.array([0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80])
    def _bitpack(nn, yy, xx, bb, ff):
        packed_data = tvm.const(0, pack_dtype)
        data_width = 8
        mask = tvm.expr.Select(bb==0, 0x1, 0x2)
        for k in range(data_width):
            f = ff * data_width + k
            element = pooled[nn, yy, xx, f]
            extracted_bit = ((element & mask) >> bb).astype(pack_dtype)
            packed_data = (packed_data | extracted_bit)
            if k < data_width - 1:
                packed_data = packed_data << 1
        return packed_data


    if pack_outputs:
        n, h, w, c = get_const_tuple(pooled.shape)
        output = tvm.compute((n, h, w, activation_bits, c//8), _bitpack, name='BitpackedConv')
    else:
        output = pooled
    return output
    
def _intrin_popcount(m, k_i, w_b, x_b, dorefa):
    dtype = 'uint8'
    w = tvm.placeholder((w_b, m, k_i), dtype=dtype, name='w')
    x = tvm.placeholder((x_b, k_i,), dtype=dtype, name='x')
    k = tvm.reduce_axis((0, k_i), name='k')
    bw = tvm.reduce_axis((0, w_b), name='bw')
    bx = tvm.reduce_axis((0, x_b), name='bx')
    if dorefa:
        z = tvm.compute((m,), lambda i:
                    tvm.sum((tvm.popcount(w[bw, i, k].astype('int16') & x[bx, k].astype('int16')) - 
                            tvm.popcount(~w[bw, i, k].astype('int16') & x[bx, k].astype('int16')))
                            << (bw+bx).astype('int16'), axis=[bw, bx, k]), name='z')
    else:
        z = tvm.compute((m,), lambda i:
                    tvm.sum(tvm.popcount(w[bw, i, k].astype('uint16') & x[bx, k].astype('uint16'))
                            << (bw+bx).astype('uint16'), axis=[bw, bx, k]), name='z')
    Wb = tvm.decl_buffer(w.shape, w.dtype,
                         name="W",
                         offset_factor=k_i,
                         strides=[tvm.var('ldw'), tvm.var('ldw'), 1]) # stride can be inferred
    Xb = tvm.decl_buffer(x.shape, x.dtype,
                         name="X",
                         offset_factor=k_i,
                         strides=[tvm.var('ldw'), 1])
    Zb = tvm.decl_buffer(z.shape, z.dtype,
                         name="Z",
                         offset_factor=1,
                         strides=[1])

    def _intrin_func(ins, outs):
        ww, xx = ins
        zz = outs[0]
        
        args_1 = tvm.const(1, 'uint32')
        args_2 = tvm.const(2, 'uint32')

        if dorefa:
            vpadd = "llvm.arm.neon.vpadd.v8i8"
            vpadalu = "llvm.arm.neon.vpadals.v16i8.v8i16"
            full_dtype = 'int8x16'
            half_dtype = 'int8x8'
            return_dtype = 'int16x8'
        else:
            vpadd = "llvm.arm.neon.vpadd.v8u8"
            vpadalu = "llvm.arm.neon.vpadalu.v16u8.v8u16"
            full_dtype = 'uint8x16'
            half_dtype = 'uint8x8'
            return_dtype = 'uint16x8'

        def _instr(index):
            irb = tvm.ir_builder.create()
            if index == 1: # reduce reset
                irb.emit(zz.vstore(0, tvm.const(0, return_dtype)))
                return irb.get()
            # body and reduce update
            cnts8 = [None] * 8
            cnts4 = [None] * 4
            cnts2 = [None] * 2
            for bw in range(w_b):
                for bx in range(x_b):
                    if k_i == 16:
                        for i in range(m):
                            w_ = ww.vload([bw, i, 0], 'uint8x16').astype(full_dtype)
                            x_ = xx.vload([bx, 0], 'uint8x16').astype(full_dtype)
                            if dorefa:
                                cnts = tvm.popcount(w_ & x_) - tvm.popcount(~w_ & x_)
                            else:
                                cnts = tvm.popcount(w_ & x_)
                            upper_half = tvm.call_pure_intrin(half_dtype, 'vectorhigh', cnts)
                            lower_half = tvm.call_pure_intrin(half_dtype, 'vectorlow', cnts)
                            cnts8[i] = upper_half + lower_half
                        for i in range(m//2):
                            cnts4[i] = tvm.call_llvm_intrin(half_dtype, vpadd,
                                                            args_1, cnts8[i*2], cnts8[i*2+1])
                        for i in range(m//4):
                            cnts2[i] = tvm.call_llvm_intrin(half_dtype, vpadd,
                                                            args_1, cnts4[i*2], cnts4[i*2+1])
                        cnts = tvm.call_pure_intrin(full_dtype, 'vectorcombine', cnts2[0], cnts2[1])
                        shifted_cnts = cnts << tvm.const(bw+bx, dtype)
                        out = tvm.call_llvm_intrin(return_dtype, vpadalu,
                                                   args_2, zz.vload(0, return_dtype), shifted_cnts)
                    else: # ki == 8
                        for i in range(m):
                            w_ = ww.vload([bw, i, 0], 'uint8x8').astype(half_dtype)
                            x_ = xx.vload([bx, 0], 'uint8x8').astype(half_dtype)
                            if dorefa:
                                cnts8[i] = tvm.popcount(w_ & x_) - tvm.popcount(~w_ & x_)
                            else:
                                cnts8[i] = tvm.popcount(w_ & x_)
                        for i in range(m//2):
                            cnts4[i] = tvm.call_llvm_intrin(half_dtype, vpadd,
                                                            args_1, cnts8[i*2], cnts8[i*2+1])
                        for i in range(m//4):
                            cnts2[i] = tvm.call_llvm_intrin(half_dtype, vpadd,
                                                            args_1, cnts4[i*2], cnts4[i*2+1])
                        cnts = tvm.call_pure_intrin(full_dtype, 'vectorcombine', cnts2[0], cnts2[1])
                        shifted_cnts = cnts << tvm.const(bw+bx, dtype)
                        out = tvm.call_llvm_intrin(return_dtype, vpadalu,
                                                   args_2, zz.vload(0, return_dtype), shifted_cnts)
                    irb.emit(zz.vstore(0, out))
            return irb.get()
        # body, reset, update
        return _instr(0), _instr(1), _instr(2)
    with tvm.build_config(offset_factor=1, partition_const_loop=True):
        return tvm.decl_tensor_intrin(z.op, _intrin_func, binds={w: Wb, x:Xb, z:Zb})

# ARM specific schedule that using custom microkernel
def _schedule_spatial_conv2d_nhwc(cfg, s, data_q, data_pad, data_vec,
                                  kernel_q, kernel_vec,
                                  conv_out, output, last, dorefa):
    return s
    _, _, _, _, _, IB, CI = data_vec.shape
    _, KH, KW, KB, _, _ = kernel_vec.shape
    KB = get_const_int(KB)
    IB = get_const_int(IB)

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]
    ##### Schedule data packing
    if data_pad is not None:
        s[data_pad].compute_inline()

    _, h, _, _, _, _, _ = s[data_vec].op.axis
    cfg.define_split("tile_ah", cfg.axis(h), policy="all", num_outputs=2, max_factor=32)
    oh, ih = cfg["tile_ah"].apply(s, data_vec, h)
    s[data_vec].parallel(oh)

    #### Schedule kernel packing
    co, _, _, _, _, _ = s[kernel_vec].op.axis
    cfg.define_split("tile_bco", cfg.axis(co), policy="all", num_outputs=2, max_factor=32)
    oco, ico = cfg["tile_bco"].apply(s, kernel_vec, co)
    s[kernel_vec].parallel(oco)

    ##### Schedule Convolution
    n, oh, ow, co, vh, vw, vc = s[conv_out].op.axis
    kh, kw, kb, ib, ci = s[conv_out].op.reduce_axis

    ci_o, ci_i = cfg['tile_ci'].apply(s, conv_out, ci)
    re_axes = cfg["reorder_0"].apply(s, conv_out, [n, oh, ow, co, vh, vw, kh, kw, ci_o, kb, ib, vc, ci_i])
    
    # Use microkernel
    kfactor = cfg['tile_ci'].size[1]
    if kfactor % 8 == 0:
        pc = _intrin_popcount(VC, kfactor, KB, IB, dorefa)
        s[conv_out].tensorize(kb, pc)  

    n, h, w, co = s[last].op.axis
    co, vc = cfg['tile_co'].apply(s, last, co)
    oh, vh = cfg['tile_oh'].apply(s, last, h)
    ow, vw = cfg['tile_ow'].apply(s, last, w)
    s[last].reorder(n, oh, ow, co, vh, vw, vc) # TODO: Is this the reorder we want
    s[last].vectorize(vc)
    if last != output:
        s[last].compute_inline()

    # oho, iho = cfg["tile_oh"].apply(s, last, oh)  # reuse parameter
    s[conv_out].compute_at(s[last], co)
    s[last].parallel(oh)
    s = s.normalize()
    return s

def _schedule_packed(cfg, s, output, pool, shift, clip, bias, conv, conv_vec, kernel_vec, data_vec, data_pad, dorefa):
    _, _, _, _, _, IB, CI = data_vec.shape
    _, KH, KW, KB, _, _ = kernel_vec.shape
    KB = get_const_int(KB)
    IB = get_const_int(IB)

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    ##### Schedule data packing
    if data_pad is not None:
        s[data_pad].compute_inline()

    # Schedule relayout
    _, h, _, _, _, _, _ = s[data_vec].op.axis
    s[data_vec].parallel(h)

    co, _, _, _, _, _ = s[kernel_vec].op.axis
    s[kernel_vec].parallel(co)

    ##### Schedule Convolution
    n, oh, ow, co, vh, vw, vc = s[conv_vec].op.axis
    kh, kw, kb, ib, ci = s[conv_vec].op.reduce_axis

    ci_o, ci_i = cfg['tile_ci'].apply(s, conv_vec, ci)
    re_axes = cfg["reorder_0"].apply(s, conv_vec, 
        [n, oh, ow, co, vh, vw, kh, kw, ci_o, kb, ib, vc, ci_i])
    
    kfactor = cfg['tile_ci'].size[1]
    pc = _intrin_popcount(VC, kfactor, KB, IB, dorefa)
    s[conv_vec].tensorize(kb, pc)  

    n_conv, h_conv, w_conv, co_conv = s[conv].op.axis
    co_conv, vc_conv = s[conv].split(co_conv, VC)
    oh_conv, ow_conv, vh_conv, vw_conv = s[conv].tile(h_conv, w_conv, VH, VW)
    s[conv].reorder(n_conv, oh_conv, ow_conv, co_conv, vh_conv, vw_conv, vc_conv)
    s[conv_vec].compute_at(s[conv], vw_conv)
    # s[conv].vectorize(vc_conv)

    # Schedule bias, clip, etc. 
    # n, h, w, co = s[bias].op.axis
    # co, vc = s[bias].split(co, VC)
    # oh, ow, vh, vw = s[bias].tile(h, w, VH, VW)
    # s[bias].reorder(n, oh, ow, co, vh, vw, vc)
    # s[conv].compute_at(s[bias], vc)
    # s[bias].compute(s[conv], vc_conv)
    # s[bias].vectorize(vc)

    # n, h, w, co = s[clip].op.axis
    # co, vc = s[clip].split(co, VC)
    # oh, ow, vh, vw = s[clip].tile(h, w, VH, VW)
    # s[clip].reorder(n, oh, ow, co, vh, vw, vc)
    # s[bias].compute_at(s[clip], vw)
    # s[clip].vectorize(vc)

    # n, h, w, co = s[shift].op.axis
    # co, vc = s[shift].split(co, VC)
    # oh, ow, vh, vw = s[shift].tile(h, w, VH, VW)
    # s[shift].reorder(n, oh, ow, co, vh, vw, vc)
    # s[clip].compute_at(s[shift], vw)
    # s[shift].vectorize(vc)

    # s[conv].compute_at(s[shift], co) <- shift is not direct consumer of conv

    # s[conv].parallel(oh_conv)
    if pool is None:
        n, oh, ow, b, co = s[output].op.axis
        s[conv].compute_at(s[output], ow)
    else:
        n, h, w, co = s[pool].op.axis
        co, vc = s[pool].split(co, VC)
        oh, ow, vh, vw = s[pool].tile(h, w, VH, VW)
        s[pool].reorder(n, oh, ow, co, vh, vw, vc)
        s[conv].compute_at(s[pool], co)
        s[pool].vectorize(vc)
        if output is None:
            output = pool
        else:
            n, oh, ow, b, co = s[output].op.axis
            s[pool].compute_at(s[output], ow)
    
    # fused = s[output].fuse(n, oh)
    s[output].parallel(oh)

    s = s.normalize()
    return s


# @generic.schedule_bitserial_conv2d_nhwc.register(["arm_cpu"])
@autotvm.register_topi_schedule(generic.nn.schedule_bitserial_conv2d_nhwc, 'arm_cpu', 'direct')
def schedule_bitserial_conv2d_nhwc_rasp(cfg, outs):
    """Raspberry pi schedule for bitserial conv2d"""
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []
    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)


        if 'spatial_bitserial_conv_nhwc' in op.tag:
            output = op.output(0)
            conv_out = op.input_tensors[0]
            kernel_vec = conv_out.op.input_tensors[0]
            kernel_q = kernel_vec.op.input_tensors[0]
            data_vec = conv_out.op.input_tensors[1]
            data_q = data_vec.op.input_tensors[0]
            data = data_q.op.input_tensors[0]
            data_pad = None
            if isinstance(data_q.op, tvm.tensor.ComputeOp) and "pad" in data_q.op.tag:
                data_pad = data_q
                data_q = data
                data = data.op.input_tensors[0]
            dorefa = "dorefa" in conv_out.op.tag
            _schedule_spatial_conv2d_nhwc(cfg, s, data_q, data_pad, data_vec,
                                          kernel_q, kernel_vec, conv_out, output, outs[0], dorefa)
        elif 'BitpackedConv' in op.name:
            # bitpacked_output = op.output(0)
            # shift = bitpacked_output.op.input_tensors[0]
            # if 'pool' in shift.op.tag:
            #     pool = shift
            #     shift = pool.op.input_tensors[0]
            # else:
            #     pool = None
            # clip = shift.op.input_tensors[0]
            # bias = clip.op.input_tensors[0]
            # conv = bias.op.input_tensors[0]
            # conv_vec = conv.op.input_tensors[0]
            # kernel_vec = conv_vec.op.input_tensors[0]
            # data_vec = conv_vec.op.input_tensors[1]
            # dorefa = "dorefa" in conv_vec.op.tag
            # _schedule_packed(cfg, s, bitpacked_output, pool, shift, clip, bias, conv, conv_vec, kernel_vec, data_vec, dorefa)

            bitpacked_output = op.output(0)
            conv = bitpacked_output.op.input_tensors[0]
            if 'pool' in conv.op.tag:
                pool = conv
                conv = pool.op.input_tensors[0]
            else:
                pool = None
            # clip = shift.op.input_tensors[0]
            # bias = clip.op.input_tensors[0]
            # conv = clip.op.input_tensors[0]
            conv_vec = conv.op.input_tensors[0]
            kernel_vec = conv_vec.op.input_tensors[0]
            data_vec = conv_vec.op.input_tensors[1]
            data_pad = data_vec.op.input_tensors[0]
            dorefa = "dorefa" in conv_vec.op.tag
            _schedule_packed(cfg, s, bitpacked_output, pool, None, None, None, conv, conv_vec, kernel_vec, data_vec, data_pad, dorefa)

        elif 'pool' in op.tag:
            pool = op.output(0)
            # shift = pool.op.input_tensors[0]
            # clip = shift.op.input_tensors[0]
            # bias = clip.op.input_tensors[0]
            conv = pool.op.input_tensors[0]
            conv_vec = conv.op.input_tensors[0]
            kernel_vec = conv_vec.op.input_tensors[0]
            data_vec = conv_vec.op.input_tensors[1]
            data_pad = data_vec.op.input_tensors[0]
            dorefa = "dorefa" in conv_vec.op.tag
            _schedule_packed(cfg, s, None, pool, None, None, None, conv, conv_vec, kernel_vec, data_vec, data_pad, dorefa)
        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s

