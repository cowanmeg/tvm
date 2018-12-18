"""TVM operator bitserial fully connected compute."""
from __future__ import absolute_import
import tvm
from tvm import autotvm
from .. import tag
from topi.util import get_const_tuple, get_const_int
from .bitserial_conv2d import bitpack # Pull out into a utility function?

# @tvm.target.override_native_generic_func("bitserial_dense")
@tvm.target.generic_func
def bitserial_dense(data, weight, data_bits, weight_bits, pack_dtype, out_dtype, dorefa):


    data_packed = bitpack(data, data_bits, pack_axis=1, bit_axis=1, pack_type=pack_dtype)
    if len(weight.shape) == 4:
        weight_packed = bitpack(weight, weight_bits, pack_axis=1, bit_axis=1, pack_type=pack_dtype)
    else:
        weight_packed = weight
    Y, DB, K = get_const_tuple(data_packed.shape)
    X, WB, _ = get_const_tuple(weight_packed.shape)

    oshape  = (Y, X)
    k = tvm.reduce_axis((0, K), name='k')
    db = tvm.reduce_axis((0, DB), name='db')
    wb = tvm.reduce_axis((0, WB), name='wb')

    matmul_dorefa = tvm.compute(oshape, 
        lambda i, j: tvm.sum( 
                (tvm.popcount(weight_packed[j, wb, k].astype(out_dtype) & data_packed[i, db, k].astype(out_dtype)) - 
                    tvm.popcount(~weight_packed[j, wb, k].astype(out_dtype) & data_packed[i, db, k].astype(out_dtype)))
                    << (db+wb).astype(out_dtype), axis=[wb, db, k]), 
        tag='bitserial_dense_dorefa')

    matmul = tvm.compute(oshape, 
                         lambda i, j: tvm.sum( 
                                tvm.popcount(weight_packed[j, wb, k].astype(out_dtype) & data_packed[i, db, k].astype(out_dtype)
                                 << (db+wb)).astype(out_dtype), axis=[wb, db, k]), 
                          tag='bitserial_dense')
                          
    if dorefa:
        return matmul_dorefa
    return matmul



@autotvm.register_topi_compute(bitserial_dense, ['cpu'], 'direct')
def bitserial_dense_generic(cfg, data, weight, data_bits, weight_bits, pack_dtype, out_dtype, dorefa):
    """The default implementation of bitserial dense in topi.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """

    data_packed = bitpack(data, data_bits, pack_axis=1, bit_axis=1, pack_type=pack_dtype)
    weight_packed = bitpack(weight, weight_bits, pack_axis=1, bit_axis=1, pack_type=pack_dtype)
    Y, DB, K = get_const_tuple(data_packed.shape)
    X, WB, _ = get_const_tuple(weight_packed.shape)

    ######## Search space
    x, y = cfg.axis(X), cfg.axis(Y)
    db, wb, k = cfg.reduce_axis(DB), cfg.reduce_axis(WB), cfg.reduce_axis(K)
    ko, ki = cfg.define_split('tile_k', k, policy='all', num_outputs=2)
    yo, yi = cfg.define_split('tile_y', y, policy='all', num_outputs=2)
    xo, xi = cfg.define_split('tile_x', x, policy='all', num_outputs=2)

    cfg.define_reorder('reorder_0', [yo, xo, ko, yi, wb, db, ki, xi], 
                        policy='candidate', candidate=[
                            [yo, xo, ko, yi, wb, db, ki, xi], 
                            [yo, xo, yi, ko, wb, db, ki, xi]
                        ])

    cfg.define_annotate('ann_reduce', [db, wb], policy='try_unroll')
    cfg.define_annotate('ann_spatial', [yi, xi], policy='try_unroll_vec')


    ###### Compute rule
    VX = cfg['tile_x'].size[-1]

    wvshape = (X//VX, WB, VX, K)
    oshape  = (Y, X)

    k = tvm.reduce_axis((0, K), name='k')
    db = tvm.reduce_axis((0, DB), name='db')
    wb = tvm.reduce_axis((0, WB), name='wb')

    # Tile data and weights
    weight_vec = tvm.compute(wvshape, lambda xo, wb, vx, k:
        weight_packed[xo*VX+vx][wb][k], name='weight_vec')

    matmul_dorefa = tvm.compute(oshape, 
        lambda i, j: tvm.sum( 
                (tvm.popcount(weight_vec[j/VX, wb, j%VX, k].astype(out_dtype) & data_packed[i, db, k].astype(out_dtype)) - 
                    tvm.popcount(~weight_vec[j/VX, wb, j%VX, k].astype(out_dtype) & data_packed[i, db, k].astype(out_dtype)))
                    << (db+wb).astype(out_dtype), axis=[wb, db, k]), 
        tag='bitserial_dense_dorefa')

    matmul = tvm.compute(oshape, 
                         lambda i, j: tvm.sum( 
                                tvm.popcount(weight_vec[j/VX, wb, j%VX, k].astype(out_dtype) & data_packed[i, db, k].astype(out_dtype)
                                 << (db+wb)).astype(out_dtype), axis=[wb, db, k]), 
                          tag='bitserial_dense')

    if (pack_dtype == 'uint8'):
        binary_op_multiplier = 8
    elif (pack_dtype == 'uint16'):
        binary_op_multiplier = 16
    elif (pack_dtype == 'uint32'):
        binary_op_multiplier = 32
    elif (pack_dtype == 'uint64'):
        binary_op_multiplier = 64

    cfg.add_flop(Y * X * K * binary_op_multiplier)

    if dorefa:
        return matmul_dorefa
    return matmul
