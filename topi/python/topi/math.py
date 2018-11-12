"""Elementwise operators"""
# pylint: disable=redefined-builtin
from __future__ import absolute_import as _abs
import tvm
from . import tag
from .util import get_const_int

@tvm.tag_scope(tag=tag.ELEMWISE)
def identity(x):
    """Take identity of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    # pylint: disable=unnecessary-lambda
    return tvm.compute(x.shape, lambda *i: x(*i))


@tvm.tag_scope(tag=tag.ELEMWISE)
def negative(x):
    """Take negation of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    # pylint: disable=unnecessary-lambda
    return tvm.compute(x.shape, lambda *i: -x(*i))


@tvm.tag_scope(tag=tag.ELEMWISE)
def exp(x):
    """Take exponential of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.exp(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def tanh(x):
    """Take hyperbolic tanh of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.tanh(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def floor(x):
    """Take floor of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.floor(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def ceil(x):
    """Take ceil of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.ceil(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def trunc(x):
    """Take truncated value of the input of x, element-wise.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.trunc(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def abs(x):
    """Take absolute value of the input of x, element-wise.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.abs(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def round(x):
    """Round elements of x to nearest integer.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.round(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def log(x):
    """Take logarithm of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.log(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def sqrt(x):
    """Take square root of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.sqrt(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def sigmoid(x):
    """Take sigmoid tanh of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.sigmoid(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def left_shift(x, n):
    """Take n bits left shift of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.
    n : int
        Number of bits.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: x(*i) << n)

@tvm.tag_scope(tag=tag.BROADCAST)
def right_left_channelwise(x, n, axis=1):
    """Take n bits left shift of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.
    n : tvm.Tensor
        Channelised number of bits.
    axis : in
        The axis where the channel data needs to be applied

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    assert len(n.shape) == 1
    assert axis < len(x.shape)
    assert get_const_int(n.shape[0]) == get_const_int(x.shape[axis])

    def _compute_channelwise(*indices):
        return x(*indices) << n(indices[axis])
    return tvm.compute(x.shape, _compute_channelwise)

@tvm.tag_scope(tag=tag.ELEMWISE)
def right_shift(x, n):
    """Take n bits right shift of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.
    n : int
        Number of bits.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: x(*i) >> n)

@tvm.tag_scope(tag=tag.BROADCAST)
def right_shift_channelwise(x, n, axis=1):
    """Take n bits right shift of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.
    n : tvm.Tensor
        Channelised number of bits.
    axis : in
        The axis where the channel data needs to be applied

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    assert len(n.shape) == 1
    assert axis < len(x.shape)
    assert get_const_int(n.shape[0]) == get_const_int(x.shape[axis])

    def _compute_channelwise(*indices):
        return x(*indices) >> n(indices[axis])
    return tvm.compute(x.shape, _compute_channelwise)


@tvm.tag_scope(tag=tag.ELEMWISE)
def clip(x, a_min, a_max):
    """Clip (limit) the values in an array. Given an interval, values
    outside the interval are clipped to the interval edges.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.
    a_min : int or float
        Minimum value.
    a_max : int or float
        Maximum value.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    def _compute(*indices):
        value = x(*indices)
        const_min = tvm.const(a_min, value.dtype)
        const_max = tvm.const(a_max, value.dtype)
        return tvm.max(tvm.min(value, const_max), const_min)
    return tvm.compute(x.shape, _compute)

@tvm.tag_scope(tag=tag.BROADCAST)
def clip_channelwise(x, a_min, a_max, axis=1):
    """Clip (limit) the values in an array. Given an interval, values
    outside the interval are clipped to the interval edges.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.
    a_min : tvm.Tensor
        Channelised ,inimum value.
    a_max : tvm.Tensor
        Channelised maximum value.
    axis : in
        The axis where the channel data needs to be applied

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    assert len(a_min.shape) == 1 and len(a_max.shape) == 1
    assert axis < len(x.shape)
    assert get_const_int(a_max.shape[0]) == get_const_int(x.shape[axis])
    assert get_const_int(a_min.shape[0]) == get_const_int(x.shape[axis])

    def _compute_channelwise(*indices):
        value = x(*indices)
        const_min = a_min(indices[axis])
        const_max = a_max(indices[axis])
        return tvm.max(tvm.min(value, const_max), const_min)
    return tvm.compute(x.shape, _compute_channelwise)


def cast(x, dtype):
    """Cast input to specified data type.

    Parameters
    ----------
    x : tvm.Tensor or Expr
        Input argument.

    dtype : str
        Data type.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    if isinstance(x, tvm.tensor.Tensor):
        return tvm.compute(
            x.shape, lambda *i: x(*i).astype(dtype), tag=tag.ELEMWISE)
    return tvm.make.static_cast(dtype, x)
