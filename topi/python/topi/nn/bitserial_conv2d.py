# pylint: disable=invalid-name, unused-variable, too-many-locals, too-many-arguments, unused-argument
"""Bitserial Conv2D operators"""
from __future__ import absolute_import as _abs
from collections import namedtuple
import numpy as np
import tvm
from tvm import autotvm
from topi.transform import concatenate
from .pad import pad
from .util import get_pad_tuple
from ..util import get_const_tuple, get_const_int

# workload description of conv2d
Workload = namedtuple('Workload',
                      ['in_dtype', 'out_dtype', 'height', 'width', 'in_filter', 'out_filter',
                       'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])

SpatialPackNCHW = namedtuple('SpatialPack',
                             ['vh', 'vw', 'vc', 'ba', 'bc'])

SpatialPackNHWC = namedtuple('SpatialPack',
                             ['vh', 'vw', 'vc', 'ba', 'bc'])

# _WORKLOADS = [
#     # workloads of resnet18 on imagenet
#     # input_size, input_size, ic, oc, kh, kw, pad, pad, stride, stride
#     Workload('uint32', 'int32', 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
#     Workload('uint32', 'int32', 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
#     Workload('uint32', 'int32', 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
#     Workload('uint32', 'int32', 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
#     Workload('uint32', 'int32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
#     Workload('uint32', 'int32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
#     Workload('uint32', 'int32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
#     Workload('uint32', 'int32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
#     Workload('uint32', 'int32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
#     Workload('uint32', 'int32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
#     Workload('uint32', 'int32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),

#     # workload of alexnet on cifar10
#     Workload('int32', 'int32', 27, 27, 96, 192, 5, 5, 2, 2, 1, 1),
#     Workload('int32', 'int32', 13, 13, 192, 384, 3, 3, 1, 1, 1, 1),
#     Workload('int32', 'int32', 13, 13, 384, 384, 3, 3, 1, 1, 1, 1),
#     Workload('int32', 'int32', 13, 13, 384, 256, 3, 3, 1, 1, 1, 1),
# ]

@tvm.target.generic_func
def bitserial_conv2d_nchw(data, kernel, stride, padding, activation_bits, weight_bits,
                          pack_dtype, out_dtype, dorefa):
    """Bitserial Conv2D operator.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width] or
                       [batch, in_height, in_width, in_channel]

    filter : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width] or
		       [filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    activation_bits: int
        number of bits used for activations/input elements

    weight_bits: int
        number of bits used for weight elements

    out_dtype: str
        return type of convolution

    pack_dtype: str
        bit packing type

    dorefa: bool
        preform the bitserial dot-product using 2 popcounts (required for DoReFa-Net)

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width] or
                       [batch, out_height, out_width, out_channel]
    """
    assert isinstance(stride, int) or len(stride) == 2
    Input_q = bitpack(data, activation_bits, pack_axis=1, bit_axis=2, pack_type=pack_dtype)
    Filter_q = bitpack(filter, weight_bits, pack_axis=1, bit_axis=4, pack_type=pack_dtype)
    batch, in_channel, activation_bits, in_height, in_width = Input_q.shape
    num_filter, channel, kernel_h, kernel_w, weight_bits = Filter_q.shape

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (kernel_h, kernel_w))
    pad_before = [0, 0, 0, pad_top, pad_left]
    pad_after = [0, 0, 0, pad_down, pad_right]

    PadInput_q = pad(Input_q, pad_before, pad_after, name="pad_temp")
    # compute the output shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    out_channel = num_filter
    out_height = (in_height - kernel_h + pad_top + pad_down) // stride_h + 1
    out_width = (in_width - kernel_w + pad_left + pad_right) // stride_w + 1

    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    b1 = tvm.reduce_axis((0, activation_bits), name='b1')
    b2 = tvm.reduce_axis((0, weight_bits), name='b2')

    def _conv(nn, ff, yy, xx):
        b1b2 = (b1+b2).astype(out_dtype)
        return tvm.sum((tvm.popcount(
            PadInput_q[nn, rc, b1, yy * stride_h + ry, xx * stride_w + rx] &
            Filter_q[ff, rc, ry, rx, b2])<< (b1b2)).astype(out_dtype),
                       axis=[rc, ry, rx, b2, b1]).astype(out_dtype)

    return tvm.compute((batch, out_channel, out_height, out_width), _conv,
        name="Conv2dOutput", tag="bitserial_conv2d_nchw")

@tvm.target.generic_func
def bitserial_conv2d_nhwc(data, kernel, stride, padding, activation_bits, weight_bits,
                          pack_dtype, out_dtype, dorefa):
    """Bitserial Conv2D operator.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width] or
                       [batch, in_height, in_width, in_channel]

    filter : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width] or
		       [filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    activation_bits: int
        number of bits used for activations/input elements

    weight_bits: int
        number of bits used for weight elements

    out_dtype: str
        return type of convolution

    pack_dtype: str
        bit packing type

    dorefa: bool
        preform the bitserial dot-product using 2 popcounts (required for DoReFa-Net)

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width] or
                       [batch, out_height, out_width, out_channel]
    """
    assert isinstance(stride, int) or len(stride) == 2
    Input_q = bitpack(data, activation_bits, pack_axis=3, bit_axis=4, pack_type=pack_dtype)
    if len(kernel.shape) == 4:
        Filter_q = bitpack(kernel, weight_bits, pack_axis=2, bit_axis=4, pack_type=pack_dtype)
        kernel_h, kernel_w, _, num_filter, _ = get_const_tuple(Filter_q.shape)
    else:
        Filter_q = kernel
        kernel_h, kernel_w, _, _, num_filter = get_const_tuple(Filter_q.shape)
    batch, in_height, in_width, in_channel_q, _ = get_const_tuple(Input_q.shape)

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    pad_top, pad_left, pad_down, pad_right = padding
    # compute the output shape
    out_channel = num_filter
    out_height = (in_height - kernel_h + pad_top + pad_down) // stride_h + 1
    out_width = (in_width - kernel_w + pad_left + pad_right) // stride_w + 1
    pad_before = [0, pad_top, pad_left, 0, 0]
    pad_after = [0, pad_down, pad_right, 0, 0]
    PadInput_q = pad(Input_q, pad_before, pad_after, name="PaddedInput")

    rc = tvm.reduce_axis((0, in_channel_q), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    b1 = tvm.reduce_axis((0, activation_bits), name='b1')
    b2 = tvm.reduce_axis((0, weight_bits), name='b2')
    def _conv(nn, yy, xx, ff):
        b1b2 = (b1+b2).astype(out_dtype)
        return tvm.sum((tvm.popcount(
            PadInput_q[nn, yy * stride_h + ry, xx * stride_w + rx, rc, b1] &
            Filter_q[ry, rx, rc, ff, b2]) << b1b2).astype(out_dtype),
                       axis=[rc, ry, rx, b2, b1])

    return tvm.compute((batch, out_height, out_width, out_channel), _conv,
        name="Conv2dOutput", tag="bitserial_conv2d_nhwc")


# def _get_workload(data, kernel, stride, padding, out_dtype, layout):
#     """ Get the workload structure. """
#     assert layout == "NCHW" or layout == "NHWC", \
#         "Only support layouts NCHW and NHWC"
#     if layout == "NCHW":
#         _, CI, IH, IW = [x.value for x in data.shape]
#         CO, _, KH, KW = [x.value for x in kernel.shape]
#     else: # NHWC
#         IH, IW = data.shape[1].value, data.shape[2].value
#         KH, KW, CI, CO = [x for x in get_const_tuple(kernel.shape)]

#     HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)
#     if isinstance(stride, (tuple, list)):
#         HSTR, WSTR = stride
#     else:
#         HSTR, WSTR = stride, stride

#     return Workload(data.dtype, out_dtype, IH, IW, CI, CO, KH, KW, HPAD, WPAD, HSTR, WSTR)

# @tvm.target.generic_func
# def _get_schedule(wkl, layout):
#     # pylint: disable=unreachable
#     """ Get the platform specific schedule. """
#     target = tvm.target.current_target()
#     raise RuntimeError(
#         "No schedule for current target:{}".format(target))
#     # This return has no use, merely to supress pylint warning
#     return wkl

@autotvm.register_topi_compute(bitserial_conv2d_nchw, ['cpu', 'arm_cpu'], 'direct')
def spatial_pack_nchw(cfg, data, kernel, stride, padding, in_bits, weight_bits,
                      pack_dtype, out_dtype, dorefa):
    """ Compute convolution with pack on spatial axes. """
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    data_q = bitpack(data, in_bits, pack_axis=1, bit_axis=0, pack_type=pack_dtype)
    # Check if kernel is already prepacked
    if len(kernel.shape) == 4:
        kernel_q = bitpack(kernel, weight_bits, pack_axis=1, bit_axis=0, pack_type=pack_dtype)
        KB, CO, _, KH, KW = get_const_tuple(kernel_q.shape)
    else:
        kernel_vec = kernel
        OCO, _, KH, KW, KB, VC = get_const_tuple(kernel_vec.shape)
        CO = OCO * VC

    IB, N, CI, H, W = get_const_tuple(data_q.shape)
    KB, CO, _, KH, KW = get_const_tuple(kernel_q.shape)
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)

    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    HCAT, WCAT = KH-1, KW-1

    TH = H + 2*HPAD
    TW = W + 2*WPAD
    OH = (H + 2*HPAD - KH) // HSTR + 1
    OW = (W + 2*WPAD - KW) // WSTR + 1

    dshape = (IB, 1, CI, H, W)
    dpshape = (IB, 1, CI, TH, TW)
    kshape = (KB, CO, CI, KH, KW)
    oshape = (1, CO, OH, OW)

     # ==================== define configuration space ====================
    n, co, oh, ow = cfg.axis(N), cfg.axis(CO), cfg.axis(OH), cfg.axis(OW)
    ci, kh, kw = cfg.reduce_axis(CI), cfg.reduce_axis(KH), cfg.reduce_axis(KW)
    ib, kb = cfg.reduce_axis(in_bits), cfg.reduce_axis(weight_bits)

    co, vc = cfg.define_split('tile_co', co, policy='all', num_outputs=2,
                       filter=lambda x: max(x.size[1:]) <= 16)
    oh, vh = cfg.define_split('tile_oh', oh, policy='all', num_outputs=2,
                       filter=lambda x: max(x.size[1:]) <= 16)
    ow, vw = cfg.define_split('tile_ow', ow, policy='all', num_outputs=2,
                       filter=lambda x: max(x.size[1:]) <= 16)
    cfg.define_annotate('ann_reduce', [ib, kb, kh, kw], policy='try_unroll')

    re_axes = cfg.define_reorder("reorder_0",
                          [n, co, oh, ow, vc, vh, vw, kh, kw, kb, ib, ci],
                          policy='interval_all', interval=(6, 11))
    cfg.add_flop(2 * N * OH * OW * CO * CI * 8 * KH * KW)

    if cfg.is_fallback:
        assert True, "Error: Fall back not implemented yet"
    # ====================

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    TH = H + 2*HPAD
    TW = W + 2*WPAD
    OH = (H + 2*HPAD - KH) // HSTR + 1
    OW = (W + 2*WPAD - KW) // WSTR + 1

    dvshape = (1, TH//(VH*HSTR), TW//(VW*WSTR), CI, VH*HSTR+HCAT, VW*WSTR+WCAT, IB)
    kvshape = (CO//VC, CI, KH, KW, KB, VC)
    ovshape = (1, CO//VC, OH//VH, OW//VW, VH, VW, VC)

    DOPAD = (HPAD != 0 and WPAD != 0)
    if DOPAD:
        data_pad = pad(data_q, (0, 0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data_q

    data_vec = tvm.compute(dvshape, lambda n, h, w, ci, vh, vw, b: \
        data_pad[b][n][ci][h*VH*HSTR+vh][w*VW*WSTR+vw], name='data_vec')

    if len(kernel.shape) == 4:
        kernel_vec = tvm.compute(kvshape, lambda co, ci, dh, dw, b, vc: \
            kernel_q[b][co*VC+vc][ci][dh][dw], name='kernel_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    dh = tvm.reduce_axis((0, KH), name='dh')
    dw = tvm.reduce_axis((0, KW), name='dw')
    b1 = tvm.reduce_axis((0, IB), name='ib')
    b2 = tvm.reduce_axis((0, KB), name='kb')

    def _conv(n, co, h, w, vh, vw, vc):
        b1b2 = (b1+b2).astype(out_dtype)
        if dorefa:
            return tvm.sum((tvm.popcount(
                data_vec[n, h, w, ci, vh*HSTR+dh, vw*WSTR+dw, b1].astype(out_dtype) &
                kernel_vec[co, ci, dh, dw, b2, vc].astype(out_dtype))  -
                            tvm.popcount(
                                data_vec[n, h, w, ci, vh*HSTR+dh, vw*WSTR+dw, b1].astype(out_dtype)
                                & ~kernel_vec[co, ci, dh, dw, b2, vc]).astype(out_dtype)) << b1b2,
                           axis=[ci, dh, dw, b1, b2])

        return tvm.sum((tvm.popcount(
            data_vec[n, h, w, ci, vh*HSTR+dh, vw*WSTR+dw, b1] &
            kernel_vec[co, ci, dh, dw, b2, vc])).astype(out_dtype) << b1b2,
                       axis=[ci, dh, dw, b1, b2])

    conv = tvm.compute(ovshape, _conv, name='conv_out')

    return tvm.compute(oshape, lambda n, co, h, w:
                       conv[n][co//VC][h//VH][w//VW][h%VH][w%VW][co%VC],
                       name='conv_vec', tag='spatial_bitserial_conv_nchw')

def _kernel_vec_spatial_pack_nhwc(kernel, kernel_bits, VC, pack_dtype):
    kernel_q = bitpack(kernel, weight_bits, pack_axis=2, bit_axis=4, pack_type=pack_dtype)
    KH, KW, _, CO, KB = get_const_tuple(kernel_q.shape)
    kvshape = (CO, KH, KW, CI, VC, KB)
    return tvm.compute(kvshape, lambda co, dh, dw, ci, vc, b: \
            kernel_q[dh][dw][ci][co*VC+vc][b], name='kernel_vec')

@autotvm.register_topi_compute(bitserial_conv2d_nhwc, 'cpu', 'direct')
def spatial_pack_nhwc(cfg, data, kernel, stride, padding, in_bits, weight_bits,
                      pack_dtype, out_dtype, dorefa):
    """ Compute convolution with pack on spatial axes. """
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    data_q = bitpack(data, in_bits, pack_axis=3, bit_axis=4, pack_type=pack_dtype)
    pack_kernel = len(kernel.shape) == 4

    if pack_kernel:
        kernel_q = bitpack(kernel, weight_bits, pack_axis=2, bit_axis=4, pack_type=pack_dtype)
    else:
        kernel_q = kernel

    KH, KW, _, CO, KB = get_const_tuple(kernel_q.shape)
    N, H, W, CI, IB = get_const_tuple(data_q.shape)
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
    ci, kh, kw = cfg.reduce_axis(CI), cfg.reduce_axis(KH), cfg.reduce_axis(KW)
    ib, kb = cfg.reduce_axis(in_bits), cfg.reduce_axis(weight_bits)

    co, vc = cfg.define_split('tile_co', co, policy='all', num_outputs=2,
                       filter=lambda x: max(x.size[1:]) <= 16)
    oh, vh = cfg.define_split('tile_oh', oh, policy='all', num_outputs=2,
                       filter=lambda x: max(x.size[1:]) <= 16)
    ow, vw = cfg.define_split('tile_ow', ow, policy='all', num_outputs=2,
                       filter=lambda x: max(x.size[1:]) <= 16)
    cfg.define_annotate('ann_reduce', [ib, kb, kh, kw], policy='try_unroll')
    # TODO: check this reorder interval
    re_axes = cfg.define_reorder("reorder_0",
                          [n, oh, ow, co, vh, vw, kh, kw, kb, ib, vc, ci],
                          policy='interval_all', interval=(3, 7))
    cfg.add_flop(2 * N * OH * OW * CO * CI * 8 * KH * KW)

    if cfg.is_fallback:
        assert True, "Error: Fall back not implemented yet"
    # ====================

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    dvshape = (1, PAD_H//(VH*HSTR), PAD_W//(VW*WSTR), VH*HSTR+HCAT, VW*WSTR+WCAT, CI, IB)
    kvshape = (CO, KH, KW, CI, VC, KB)
    ovshape = (1, OH, OW, CO, VH, VW, VC)
    oshape = (1, OH, OW, CO)

    if (DPAD != 0 and RPAD != 0):
        data_pad = pad(data_q, (0, TPAD, LPAD, 0, 0), (0, DPAD, RPAD, 0, 0), name="data_pad")
    else:
        data_pad = data_q

    data_vec = tvm.compute(dvshape, lambda n, h, w, vh, vw, ci, b: \
        data_pad[n][h*VH*HSTR+vh][w*VW*WSTR+vw][ci][b], name='data_vec')

    kernel_vec = tvm.compute(kvshape, lambda co, dh, dw, ci, vc, b: \
        kernel_q[dh][dw][ci][co*VC+vc][b], name='kernel_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    dh = tvm.reduce_axis((0, KH), name='dh')
    dw = tvm.reduce_axis((0, KW), name='dw')
    b1 = tvm.reduce_axis((0, IB), name='ib')
    b2 = tvm.reduce_axis((0, KB), name='kb')

    def _conv(n, h, w, co, vh, vw, vc):
        b1b2 = (b1+b2).astype(out_dtype)
        if dorefa:
            return tvm.sum(
                ((tvm.popcount(data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ci, b1]&
                              kernel_vec[co, dh, dw, ci, vc, b2]).astype(out_dtype) -
                 tvm.popcount(data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ci, b1]&
                              ~kernel_vec[co, dh, dw, ci, vc, b2]).astype(out_dtype)) << b1b2),
                axis=[dh, dw, ci, b1, b2])

        return tvm.sum(tvm.popcount(
            data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ci, b1] &
            kernel_vec[co, dh, dw, ci, vc, b2]).astype(out_dtype) << b1b2,
                       axis=[dh, dw, ci, b1, b2])

    conv = tvm.compute(ovshape, _conv, name='conv')

    return tvm.compute(oshape, lambda n, h, w, co:
                       conv[n][h//VH][w//VW][co//VC][h%VH][w%VW][co%VC],
                       name='output_unpack', tag='spatial_bitserial_conv_nhwc')

def bitpack(data, bits, pack_axis, bit_axis, pack_type, name="QuantizeInput"):
    """Packs data into format necessary for bitserial computation
    pack_axis : int
       index of the axis to pack in data
    bit_axis : int
       index of axis to place bit axis in resulting packed data"""
    ishape = data.shape
    n = len(ishape)
    if pack_type == 'uint8':
        data_width = 8
    elif pack_type == 'uint16':
        data_width = 16
    elif pack_type == 'uint32':
        data_width = 32
    elif pack_type == 'uint64':
        data_width = 64

    # Data must be in multiples of the data_width
    assert get_const_int(ishape[pack_axis]) % data_width == 0, "Not a multiple of word size"

    shape_vec = list(ishape)
    shape_vec[pack_axis] = (shape_vec[pack_axis] // data_width)
    shape_vec.insert(bit_axis, 1)
    bitserial_oshape = tuple(shape_vec)
    masks = np.array([0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80])

    # pack axis shifts if bit axis comes before
    if bit_axis <= pack_axis:
        pack_axis += 1

    def _bitpack(*indices):
        packed_data = [tvm.const(0, pack_type)] * bits
        for k in range(data_width):
            # Translate indices for packed data back to original
            idx = [0] * n
            j = 0
            for i in range(n+1):
                if i == bit_axis:
                    continue
                elif i == pack_axis:
                    idx[j] = indices[i] * data_width + k
                else:
                    idx[j] = indices[i]
                j += 1

            element = data(*idx)
            for b in range(bits):
                extracted_bit = ((element & tvm.const(masks[b])) >> b).astype(pack_type)
                packed_data[b] = (packed_data[b] | extracted_bit)
                if k < data_width - 1:
                    packed_data[b] = packed_data[b] << 1

            if k == data_width - 1:
                return tuple(packed_data)
        return tuple(packed_data)

    output_tuple = tvm.compute(bitserial_oshape, _bitpack, name=name, tag='bitpack')

    if bits > 1:
        return concatenate(output_tuple, axis=bit_axis)
    return output_tuple

_SCH_TO_DECL_FUNC_QUANT = {
    SpatialPackNCHW: spatial_pack_nchw,
    SpatialPackNHWC: spatial_pack_nhwc,
}
