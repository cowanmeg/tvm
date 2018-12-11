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
# from ..nn.bitserial_conv2d import SpatialPackNCHW, _WORKLOADS, spatial_pack_nchw
from ..nn.util import get_pad_tuple
from ..util import get_const_int, get_const_tuple
from .. import generic
from tvm.autotvm.task.nnvm_integration import deserialize_args

RaspSpatialPack = namedtuple('SpatialPack',
                             ['vh', 'vw', 'vc', 'ba', 'bc', 'split_ci', 'kfactor'])

# _QUANTIZED_SCHEDULES_NHWC = [
#     RaspSpatialPack(2, 2, 8, 1, 1, False, 8),
#     RaspSpatialPack(1, 4, 8, 4, 1, False, 8),
#     RaspSpatialPack(1, 4, 8, 1, 16, False, 8),
#     RaspSpatialPack(1, 4, 8, 4, 8, False, 8),
#     RaspSpatialPack(1, 7, 8, 3, 8, False, 16),
#     RaspSpatialPack(1, 2, 8, 1, 8, False, 16),
#     RaspSpatialPack(2, 1, 8, 1, 4, False, 16),
#     RaspSpatialPack(1, 7, 8, 1, 1, True, 16),
#     RaspSpatialPack(1, 1, 8, 1, 16, True, 16),
#     RaspSpatialPack(1, 1, 8, 1, 8, True, 16),
#     RaspSpatialPack(1, 1, 8, 1, 16, True, 16),
# ]

# _QUANTIZED_SCHEDULES_NCHW = [
#     # resnet
#     SpatialPackNCHW(2, 2, 8, 1, 1),
#     SpatialPackNCHW(1, 4, 8, 4, 1),
#     SpatialPackNCHW(1, 4, 8, 1, 16),
#     SpatialPackNCHW(1, 4, 8, 4, 8),
#     SpatialPackNCHW(1, 7, 8, 3, 8),
#     SpatialPackNCHW(1, 2, 8, 1, 8),
#     SpatialPackNCHW(2, 1, 8, 1, 4),
#     SpatialPackNCHW(1, 7, 8, 1, 1),
#     SpatialPackNCHW(1, 1, 8, 1, 16),
#     SpatialPackNCHW(1, 1, 8, 1, 8),
#     SpatialPackNCHW(1, 1, 8, 1, 16),
# ]

@autotvm.task.register("topi_nn_bitserial_conv2d_nhwc")
def _topi_bitserial_conv2d(*args, **kwargs):
    args = deserialize_args(args)
    C = topi.nn.bitserial_conv2d_nhwc(*args, **kwargs)
    s = generic.nn.schedule_bitserial_conv2d_nhwc([C])
    data = args[0]
    kernel = args[1]
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


# @bitserial_conv2d_nhwc.register("arm_cpu")
# def bitserial_conv2d_arm_cpu_arg_to_workload(data, kernel, stride, padding, activation_bits, weight_bits,
#                      pack_dtype=None, out_dtype=None, dorefa=False):
#     if out_dtype is None:
#         out_dtype = data.dtype
#     assert data.shape[0].value == 1, "only support batch size=1 convolution on rasp"
#     assert layout == "NCHW" or layout == "NHWC", "only support layouts NCHW and NHWC"
#     if dorefa:
#         assert layout == "NCHW", "Cannot support dorea with NHWC layout yet"
#     wkl = _get_workload(data, kernel, stride, padding, out_dtype, layout)
#     # sch = _get_schedule(wkl, layout)
#     if layout == "NCHW":
#         return spatial_pack_nchw(data, kernel, stride, padding, activation_bits, weight_bits,
#                                  pack_dtype=pack_dtype, out_dtype=out_dtype, dorefa=dorefa)
#     #return spatial_pack_nhwc(data, kernel, stride, padding, activation_bits,
#                               #weight_bits, out_dtype)
#     return ('bitserial_conv2d', ) + autotvm.task.args_to_workload(
#         [data, kernel, stride, padding, activation_bits, weight_bits, out_dtype])

def _kernel_vec_spatial_pack_nhwc(kernel, kernel_bits, VC, bitpack=True):
    if bitpack:
        kernel_q = bitpack(kernel, kernel_bits, pack_axis=2, bit_axis=2, pack_type='uint8')
    else:
        kernel_q = kernel
    KH, KW, KB, CI, CO = kernel_q.shape
    kvshape = (CO//VC, KH, KW, KB, VC, CI)
    return tvm.compute(kvshape, lambda co, dh, dw, b, vc, ci: \
        kernel_q[dh][dw][b][ci][co*VC+vc], name='kernel_vec')

# TODO: support kernel prepacking
@autotvm.register_topi_compute(bitserial_conv2d_nhwc, 'arm_cpu', 'direct')
def spatial_pack_nhwc(cfg, data, kernel, stride, padding, activation_bits, weight_bits, 
                      pack_dtype, out_dtype, dorefa, shift=None):
    """ Compute convolution with pack on spatial axes. """
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    assert pack_dtype == 'uint8', "only support packing into 8 bits"
    # wkl = _get_workload(data, kernel, stride, padding, out_dtype, "NHWC")
    # sch = _get_schedule(wkl, "NHWC")
    
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
    # ==================== define configuration space ====================
    n, oh, ow, co = cfg.axis(N), cfg.axis(OH), cfg.axis(OW), cfg.axis(CO)
    ci, kh, kw = cfg.reduce_axis(CI_packed), cfg.reduce_axis(KH), cfg.reduce_axis(KW)
    ib, kb = cfg.reduce_axis(activation_bits), cfg.reduce_axis(weight_bits)

    co, vc = cfg.define_split('tile_co', co, policy='all', num_outputs=2,
                       filter=lambda x: x.size[-1] == 8)
    oh, vh = cfg.define_split('tile_oh', oh, policy='all', num_outputs=2,
                       filter=lambda x: max(x.size[1:]) <= 16)
    ow, vw = cfg.define_split('tile_ow', ow, policy='all', num_outputs=2,
                       filter=lambda x: max(x.size[1:]) <= 16)
    cfg.define_annotate('ann_reduce', [ib, kb, kh, kw], policy='try_unroll')
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

    print (VC, VH, VW, cfg['tile_ci'])

    data_q = bitpack(data, activation_bits, pack_axis=3, bit_axis=3, pack_type='uint8')
    kernel_vec = _kernel_vec_spatial_pack_nhwc(kernel, weight_bits, VC, len(kernel.shape) == 4)
    N, H, W, IB, CI = data_q.shape
    OCO, KH, KW, KB, VC, _ = kernel_vec.shape

    dvshape = (N, PAD_H//(VH*HSTR), PAD_W//(VW*WSTR), VH*HSTR+HCAT, VW*WSTR+WCAT, IB, CI)
    ovshape = (1, OH // VH, OW // VW, CO // VC, VH, VW, VC)

    if (TPAD != 0 and RPAD != 0):
        data_pad = pad(data_q, (0, TPAD, LPAD, 0, 0), (0, DPAD, RPAD, 0, 0), name="data_pad")
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
        conv = tvm.compute(ovshape, _dorefa_conv, name='conv', tag='dorefa')
    else:
        conv = tvm.compute(ovshape, _conv, name='conv')


    return tvm.compute(oshape, lambda n, h, w, co:
                       conv[n][h//VH][w//VW][co//VC][h%VH][w%VW][co%VC].astype(out_dtype),
                       name='output_vec', tag='spatial_bitserial_conv_nhwc')

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
                         strides=[tvm.var('ldw'), tvm.var('ldw'), 1])
    Xb = tvm.decl_buffer(x.shape, x.dtype,
                         name="X",
                         offset_factor=k_i,
                         strides=[tvm.var('ldw'), 1])

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
            if index == 1:
                irb.emit(zz.vstore(0, tvm.const(0, return_dtype)))
                return irb.get()

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
        return tvm.decl_tensor_intrin(z.op, _intrin_func, binds={w: Wb, x:Xb})

# ARM specific schedule that using custom microkernel
def _schedule_spatial_conv2d_nhwc(cfg, s, data, data_q, data_pad, data_vec,
                                  kernel_q, kernel_vec,
                                  conv_out, output, last, dorefa):
    # no stride and padding info here
    _, H, W, IB, CI = data_q.shape
    KH, KW, KB, _, CO = kernel_q.shape
    KB = get_const_int(KB)
    IB = get_const_int(IB)

    # if data_pad is None:
    #     padding = (0, 0)
    #     _, in_h, in_w, _, _ = data_q.shape
    #     kern_h, kern_w, _, _ = kernel.shape
    #     _, out_h, out_w, _ = output.shape
    #     hstride = (in_h - kern_h) // (out_h - 1)
    #     wstride = (in_w - kern_w) // (out_w - 1)
    #     stride = get_const_int(hstride), get_const_int(wstride)
    # else:
    #     _, in_h, in_w, _, _ = data_q.shape
    #     _, pad_h, pad_w, _, _ = data_pad.shape
    #     hpad = (pad_h - in_h) // 2
    #     wpad = (pad_w - in_w) // 2
    #     padding = get_const_int(hpad), get_const_int(wpad)

    #     _, in_h, in_w, _, _ = data_pad.shape
    #     kern_h, kern_w, _, _ = kernel.shape
    #     _, out_h, out_w, _ = output.shape
    #     hstride = (in_h - kern_h) // (out_h - 1)
    #     wstride = (in_w - kern_w) // (out_w - 1)
    #     stride = get_const_int(hstride), get_const_int(wstride)

    # wkl = _get_workload(data, kernel, stride, padding, output.dtype, "NHWC")
    # sch = _get_schedule(wkl, "NHWC")

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]
    # ba = cfg.ba
    # bc = cfg.bc

    ##### Schedule data packing
    if data_pad is not None:
        s[data_pad].compute_inline()

    _, h, _, _, _, _, _ = s[data_vec].op.axis
    cfg.define_split("tile_ah", cfg.axis(h), policy="all", num_outputs=2, max_factor=32)
    oh, ih = cfg["tile_ah"].apply(s, data_vec, h)
    if cfg["tile_ah"].size[1] == 1:
        oaxis = oh
        paxis = oh
    else:
        oaxis = oh
        paxis = ih

    s[data_vec].parallel(paxis)

    #### Schedule kernel packing
    co, _, _, _, _, _ = s[kernel_vec].op.axis
    cfg.define_split("tile_bco", cfg.axis(co), policy="all", num_outputs=2, max_factor=32)
    oco, ico = cfg["tile_bco"].apply(s, kernel_vec, co)
    if cfg["tile_bco"].size[1] == 1:
        oaxis = oco
        paxis = oco
    else:
        oaxis = oco
        paxis = ico

    s[kernel_vec].parallel(paxis)

    ##### Schedule Convolution
    n, oh, ow, co, vh, vw, vc = s[conv_out].op.axis
    kh, kw, kb, ib, ci = s[conv_out].op.reduce_axis

    ci_o, ci_i = cfg['tile_ci'].apply(s, conv_out, ci)
    re_axes = cfg["reorder_0"].apply(s, conv_out, [n, oh, ow, co, vh, vw, kh, kw, ci_o, kb, ib, vc, ci_i])
    
    vc_len = cfg.axis(vc).length
    kfactor = cfg['tile_ci'].size[1]
    pc = _intrin_popcount(vc_len, kfactor, KB, IB, dorefa)
    s[conv_out].tensorize(kb, pc)  

    n, h, w, co = s[last].op.axis
    co, vc = s[last].split(co, VC)
    oh, ow, vh, vw = s[last].tile(h, w, VH, VW)
    s[last].reorder(n, oh, ow, co, vc, vh, vw)
    s[last].vectorize(vw)
    if last != output:
        s[last].compute_inline()

    oho, iho = cfg["tile_oh"].apply(s, last, oh)  # reuse parameter
    s[conv_out].compute_at(s[last], ow)
    if cfg["tile_oh"].size[1] == 1:
        oaxis = oho
        paxis = oho
    else:
        oaxis = oho
        paxis = iho

    s[last].parallel(paxis)
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
            # kernel = kernel_q.op.input_tensors[0]
            # if "QuantizeInput" in kernel.op.name:
            #     # Need to go up 1 further, from the combine in bitpack
            #     kernel = kernel.op.input_tensors[0]
            data_vec = conv_out.op.input_tensors[1]
            data_q = data_vec.op.input_tensors[0]
            data = data_q.op.input_tensors[0]
            data_pad = None
            if isinstance(data_q.op, tvm.tensor.ComputeOp) and "pad" in data_q.op.tag:
                data_pad = data_q
                data_q = data
                data = data_q.op.input_tensors[0]
            if "QuantizeInput" in data.op.name:
                # Need to go up 1 further, from the combine in bitpack
                data = data.op.input_tensors[0]
            dorefa = "dorefa" in conv_out.op.tag
            _schedule_spatial_conv2d_nhwc(cfg, s, data, data_q, data_pad, data_vec,
                                          kernel_q, kernel_vec, conv_out, output, outs[0], dorefa)
        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s
