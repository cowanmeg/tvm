"""Schedule for bitserial dense operator."""
from __future__ import absolute_import as _abs
import tvm
import topi
from tvm import autotvm
from .. import tag
from .. import generic
from .bitserial_conv2d import _intrin_popcount
from topi.util import get_const_tuple, get_const_int
from tvm.autotvm.task.nnvm_integration import deserialize_args
from ..nn.bitserial_dense import bitserial_dense
from ..nn.bitserial_conv2d import bitpack # Pull out into a utility function?


@autotvm.task.register("topi_nn_bitserial_dense")
def _topi_bitserial_dense(*args, **kwargs):
    args = deserialize_args(args)
    C = topi.nn.bitserial_dense(*args, **kwargs)
    s = generic.nn.schedule_bitserial_dense([C])
    data = args[0]
    kernel = args[1]
    return s, [data, kernel, C]

# Slightly modified dense compute rule that forces tiling to fit arm_cpu microkernel
@autotvm.register_topi_compute(bitserial_dense, ['arm_cpu'], 'direct')
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
    if len(weight.shape) == 2:
        weight_packed = bitpack(weight, weight_bits, pack_axis=1, bit_axis=1, pack_type=pack_dtype)
    else: 
        weight_packed = weight
    Y, DB, K = get_const_tuple(data_packed.shape)
    X, WB, _ = get_const_tuple(weight_packed.shape)

    # Pad Inputs so that microkernel can be used
    if (X % 8 != 0):
        X_PAD = X % 8
        data_packed = pad(data_packed, [0, 0, 0], [X_PAD, 0, 0], name='PaddedInput')
        X += X_PAD

    ######## Search space
    x, y = cfg.axis(X), cfg.axis(Y)
    db, wb, k = cfg.reduce_axis(DB), cfg.reduce_axis(WB), cfg.reduce_axis(K)
    ko, ki = cfg.define_split('tile_k', k, policy='all', num_outputs=2, 
                                filter=lambda xx: xx.size[-1] == 8 or xx.size[-1] == 16)
    yo, yi = cfg.define_split('tile_y', y, policy='all', num_outputs=2)
    
    # if (X % 8 == 0): # Can use the arm microkernel
    xo, xi = cfg.define_split('tile_x', x, policy='all', num_outputs=2,
                                    filter=lambda xx: xx.size[-1] == 8)
    # else:
    #     xo, xi = cfg.define_split('tile_x', x, policy='all', num_outputs=2)

    cfg.define_reorder('reorder_0', [yo, xo, ko, yi, wb, db, xi, ki], 
                        policy='candidate', candidate=[
                            [yo, xo, ko, yi, wb, db, xi, ki], 
                            [yo, xo, yi, ko, wb, db, xi, ki],
                            [yo, yi, xo, ko, wb, db, xi, ki]
                        ])

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
                                 << (db+wb)).astype(out_dtype), axis=[db, wb, k]), 
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


@autotvm.register_topi_schedule(generic.nn.schedule_bitserial_dense, ['arm_cpu'], 'direct')
def schedule_bitserial_dense(cfg, outs):
    """Schedule for binary_dense.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of bitserial dense operator.
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for bitserial_dense.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _schedule(cfg, s, data, weight, data_vec, weight_vec, output, dorefa):
        s[data_vec].parallel(s[data_vec].op.axis[0])
        # s[data_vec].vectorize(s[data_vec].op.axis[-1])
        s[weight_vec].parallel(s[weight_vec].op.axis[0])
        # s[weight_vec].vectorize(s[weight_vec].op.axis[-1])

        y, x = s[output].op.axis
        wb, db, k = s[output].op.reduce_axis

        Y, DB, K = get_const_tuple(data_vec.shape)
        XO, WB, VX, K = get_const_tuple(weight_vec.shape)

        yo, yi = cfg["tile_y"].apply(s, output, y)
        xo, xi = cfg["tile_x"].apply(s, output, x)
        ko, ki = cfg["tile_k"].apply(s, output, k)
        
        cfg["reorder_0"].apply(s, output, [yo, xo, ko, yi, wb, db, xi, ki])
        
        nfactor = cfg['tile_x'].size[1]
        kfactor = cfg['tile_k'].size[1]
        
        if nfactor % 8 == 0:
            pc = _intrin_popcount(nfactor, kfactor, WB, DB, dorefa)
            s[output].tensorize(wb, pc)   

        s[output].parallel(yo)
        s = s.normalize()
        return s

    def traverse(op):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag) or 'elemwise' in op.tag:
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)

        elif op.tag == 'bitserial_dense' or 'bitserial_dense_dorefa':
            output = op.output(0)
            weight_vec = op.input_tensors[0]
            weight_packed = weight_vec.op.input_tensors[0]
            # weight = weight_packed.op.input_tensors[0]
            # if "QuantizeInput" in weight.op.name:
            #     # Need to go up 1 further, from the combine in bitpack
            #     weight = weight.op.input_tensors[0]

            data_vec = op.input_tensors[1]
            # data_packed = data_vec.op.input_tensors[0]
            data = data_vec.op.input_tensors[0]
            if "QuantizeInput" in data.op.name:
                data = data.op.input_tensors[0]
            dorefa = (output.op.tag == 'bitserial_dense_dorefa')
            _schedule(cfg, s, data, weight_packed, data_vec, weight_vec, output, dorefa)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

    traverse(outs[0].op)
    return s
