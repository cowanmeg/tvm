"""Schedule for bitserial dense operator."""
from __future__ import absolute_import as _abs
import tvm
import topi
from tvm import autotvm
from .. import tag
from .. import generic
from topi.util import get_const_tuple, get_const_int
from tvm.autotvm.task.nnvm_integration import deserialize_args

@autotvm.task.register("topi_x86_bitserial_dense")
def _topi_bitserial_dense(*args, **kwargs):
    args = deserialize_args(args)
    C = topi.nn.bitserial_dense(*args, **kwargs)
    s = generic.nn.schedule_bitserial_dense([C])
    data = args[0]
    kernel = args[1]
    return s, [data, kernel, C]


# @generic.schedule_bitserial_dense.register(["cpu"])
@autotvm.register_topi_schedule(generic.nn.schedule_bitserial_dense, ['cpu'], 'direct')
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

    def _schedule(cfg, s, data, weight, data_vec, weight_vec, output):
        s[data_vec].parallel(s[data_vec].op.axis[0])
        s[weight_vec].parallel(s[weight_vec].op.axis[0])

        y, x = s[output].op.axis
        wb, db, k = s[output].op.reduce_axis

        Y, DB, K = get_const_tuple(data_vec.shape)
        XO, WB, VX, K = get_const_tuple(weight_vec.shape)

        yo, yi = cfg["tile_y"].apply(s, output, y)
        xo, xi = cfg["tile_x"].apply(s, output, x)
        ko, ki = cfg["tile_k"].apply(s, output, k)
        

        cfg["reorder_0"].apply(s, output, [yo, xo, ko, yi, wb, db, ki, xi])
        cfg["ann_reduce"].apply(s, output, [db, wb],
                                axis_lens=[get_const_int(db.dom.extent),
                                        get_const_int(wb.dom.extent)],
                                max_unroll=8,
                                cfg=cfg)
        cfg["ann_spatial"].apply(s, output, [yi, xi],
                                    axis_lens=[cfg['tile_y'].size[-1],
                                                cfg['tile_x'].size[-1]],
                                    max_unroll=8,
                                    cfg=cfg)
    #    s[output].vectorize()

        s[output].parallel(yo)
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
            _schedule(cfg, s, data, weight_packed, data_vec, weight_vec, output)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

    traverse(outs[0].op)
    return s
