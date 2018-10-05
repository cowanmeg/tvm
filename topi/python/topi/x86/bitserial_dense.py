"""Schedule for bitserial dense operator."""
from __future__ import absolute_import as _abs
import tvm
from .. import tag
from .. import generic

# TODO: add some tuning 
@generic.schedule_bitserial_dense.register(["cpu"])
def schedule_bitserial_dense(outs):
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

    def _schedule(data_q, weight_q, weight_vec, output):
        i, j = s[output].op.axis
        db, wb, k = s[output].op.reduce_axis
        jo, ji = s[output].split(j, factor=2)

        s[output].reorder(i, jo, k, wb, db, ji)
        s[output].vectorize(ji)
        s[output].unroll(db)
        s[output].unroll(wb)

        bc = 2
        p_axis = jo
        if bc == 1:
            oaxis = p_axis
            paxis = p_axis
        else:
            oco, ico = s[output].split(p_axis, bc)
            oaxis = oco
            paxis = ico

        s[output].parallel(paxis)
        s[output].pragma(oaxis, "parallel_launch_point")
        s[output].pragma(paxis, "parallel_stride_pattern")
        s[output].pragma(oaxis, "parallel_barrier_when_finish")


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
            _schedule(data_q, weight_q, weight_vec, output)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

    traverse(outs[0].op)
    return s
