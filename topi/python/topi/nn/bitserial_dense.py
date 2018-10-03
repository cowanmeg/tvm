"""TVM operator bitserial fully connected compute."""
from __future__ import absolute_import
import tvm
from .. import tag
from .bitserial_conv2d import bitpack # Pull out into a utility function?

def bitserial_dense_default(data, weight, data_bits, weight_bits, out_dtype='int32'):
    """The default implementation of bitserial dense in topi.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2 and len(weight.shape) == 2, \
        "only support 2-dim dense"
    # if bias is not None:
    #     assert len(bias.shape) == 1
    data_q = bitpack(data, data_bits, pack_axis=1, bit_axis=1, pack_type='uint32')
    weight_q = bitpack(weight, weight_bits, pack_axis=1, bit_axis=1, pack_type='uint32')
    batch, _, in_dim = data_q.shape
    out_dim, _, _ = weight_q.shape
    k = tvm.reduce_axis((0, in_dim), name='k')
    db = tvm.reduce_axis((0, data_bits), name='db')
    wb = tvm.reduce_axis((0, weight_bits), name='wb')
    matmul = tvm.compute((batch, out_dim), 
                        # lambda i, j: data_q[0, 0, 0],
                         lambda i, j: tvm.sum( 
                                ((tvm.popcount(data_q[i, db, k] & weight_q[j, wb, k]) - 
                                 tvm.popcount(data_q[i, db, k] & ~weight_q[j, wb, k])) 
                                 << (db+wb)).astype(out_dtype), axis=[db, wb, k]), 
                          tag='bitserial_dense')
    # probably wouldn't put bias here?
    # if bias is not None:
    #     matmul = tvm.compute((batch, out_dim), \
    #                          lambda i, j: matmul[i, j] + bias[j], \
    #                          tag=tag.BROADCAST)
    return matmul


@tvm.target.override_native_generic_func("bitserial_dense")
def bitserial_dense(data, weight, data_bits, weight_bits, out_dtype='int32'):
    """Applies a linear transformation: :math:`Y = XW^T + b`.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    return bitserial_dense_default(data, weight, data_bits, weight_bits, out_dtype)
