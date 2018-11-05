"""Schedule for bitserial dense operator."""
from __future__ import absolute_import as _abs
import tvm
import topi
from tvm import autotvm
from .. import tag
from .. import generic
from ..nn import bitserial_dense_topi
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
@autotvm.register_topi_schedule(generic.nn.schedule_bitserial_dense, ['arm_cpu', 'cpu'], 'direct')
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

    def _schedule(cfg, s, data_q, weight_q, weight_vec, output):
        y, x = s[output].op.axis
        DB, WB, K = s[output].op.reduce_axis

        # Search space
        db, wb, k = cfg.reduce_axis(DB), cfg.reduce_axis(WB), cfg.reduce_axis(K)
        yo, yi = cfg.define_split('tile_y', y, num_outputs=2) # support 3 splits
        xo, xi = cfg.define_split('tile_x', x, num_outputs=2)
        cfg.define_reorder('reorder_0', [yo, xo, k, wb, db, yi ,xi], 
                            policy='candidate', candidate=[
                                [yo, xo, k, wb, db, yi ,xi], 
                                [yo, xo, wb, k, db, yi ,xi], 
                                [yo, xo, wb, db, k, yi ,xi],
                                [yo, xo, wb, db, yi, k, xi]
                            ])
        cfg.define_annotate('ann_reduce', [db, wb, k], policy='try_unroll')
        cfg.define_annotate('ann_spatial', [yi, xi], policy='try_unroll_vec')

        # schedule according to config
        yo, yi= cfg["tile_y"].apply(s, output, y)
        xo, xi = cfg["tile_x"].apply(s, output, x)

        cfg["reorder_0"].apply(s, output, [yo, xo, K, WB, DB, yi, xi])
        cfg["ann_reduce"].apply(s, output, [DB, WB, K],
                                axis_lens=[get_const_int(DB.dom.extent),
                                        get_const_int(WB.dom.extent),
                                        get_const_int(K.dom.extent)],
                                max_unroll=8,
                                cfg=cfg)
        cfg["ann_spatial"].apply(s, output, [yi, xi],
                                    axis_lens=[cfg['tile_y'].size[-1],
                                                cfg['tile_x'].size[-1]],
                                    max_unroll=8,
                                    cfg=cfg)

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

        elif op.tag == 'bitserial_dense':
            output = op.output(0)
            data_q = op.input_tensors[1]
            weight_vec = op.input_tensors[0]
            weight_q = op.input_tensors[0]
            # if "QuantizeInput" in data_q.op.name:
            #     # Need to go up 1 further, from the combine in bitpack
            #     data_q = data_q.op.input_tensors[0]
            # if "QuantizeInput" in weight_q.op.name:
            #     # Need to go up 1 further, from the combine in bitpack
            #     weight_q = weight_q.op.input_tensors[0]
            _schedule(cfg, s, data_q, weight_q, weight_vec, output)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

    traverse(outs[0].op)
    return s
