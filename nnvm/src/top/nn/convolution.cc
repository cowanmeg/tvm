/*!
 *  Copyright (c) 2017 by Contributors
 * \file convolution.cc
 * \brief Convolution operators
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/layout.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/nn.h>
#include <tvm/tensor.h>
#include <tvm/packed_func_ext.h>
#include <nnvm/compiler/op_attr_types.h>
#include <tvm/tvm.h>
#include "nn_common.h"
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "topi/nn.h"


using tvm::Tensor;
using tvm::Array;
using nnvm::compiler::FTVMCompute;

namespace nnvm {
namespace top {

// conv2d
DMLC_REGISTER_PARAMETER(Conv2DParam);
inline bool Conv2DInferShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_shape,
                             std::vector<TShape>* out_shape) {
  static const Layout kNCHW("NCHW");
  static const Layout kOIHW("OIHW");

  const Conv2DParam& param = nnvm::get<Conv2DParam>(attrs.parsed);

  const Layout in_layout(param.layout);
  const Layout kernel_layout(param.kernel_layout);
  CHECK(in_layout.convertible(kNCHW))
    << "Conv only support input layouts that are convertible from NCHW."
    << " But got " << in_layout;
  CHECK(kernel_layout.convertible(kOIHW))
    << "Conv only support kernel layouts that are convertible from OIHW."
    << " But got "<< kernel_layout;

  Layout out_layout(param.out_layout);
  if (!out_layout.defined()) out_layout = in_layout;
  CHECK(out_layout.convertible(kNCHW))
    << "Conv only support output layouts that are convertible from NCHW."
    << " But got " << out_layout;

  if (param.use_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_shape->size(), 1U);

  TShape dshape = in_shape->at(0);
  if (dshape.ndim() == 0) return false;
  dshape = ConvertLayout(dshape, in_layout, kNCHW);

  CHECK_EQ(dshape.ndim(), 4U) << "Input data should be 4D";
  CHECK_EQ(param.kernel_size.ndim(), 2U);
  CHECK_EQ(param.strides.ndim(), 2U)
      << "incorrect stride size: " << param.strides;
  CHECK_EQ(param.dilation.ndim(), 2U)
      << "incorrect dilate size: " << param.dilation;
  CHECK_EQ(dshape[1] % param.groups, 0U)
      << "input channels must divide group size";
  CHECK_EQ(param.channels % param.groups, 0U)
      << "output channels must divide group size";

  TShape wshape({param.channels,
                 dshape[1] / param.groups,
                 param.kernel_size[0],
                 param.kernel_size[1]});

  wshape = ConvertLayout(wshape, kOIHW, kernel_layout);

  if (in_shape->at(Conv2DParam::kWeight).ndim() == 0) {
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, Conv2DParam::kWeight, wshape);
  }
  if (param.use_bias) {
    static const Layout default_bias_layout("C");
    TShape bias_shape({param.channels});
    auto oc_block = out_layout.subsizeof('C');
    if (oc_block > 0) {
      size_t split_axis = (out_layout.indexof('C') < out_layout.indexof('c')) ? 1 : 0;
      bias_shape = ConvertLayout(bias_shape, default_bias_layout,
                                 default_bias_layout.split('C', split_axis, oc_block));
    }
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, Conv2DParam::kBias, bias_shape);
  }
  // dilation
  dim_t dilated_ksize_y = 1 + (param.kernel_size[0] - 1) * param.dilation[0];
  dim_t dilated_ksize_x = 1 + (param.kernel_size[1] - 1) * param.dilation[1];
  TShape oshape({dshape[0], param.channels, 0, 0});
  if (dshape[2] != 0) {
    oshape[2] = (dshape[2] + param.padding[0] + param.padding[2] - dilated_ksize_y) / param.strides[0] + 1;
  }
  if (dshape[3] != 0) {
    oshape[3] = (dshape[3] + param.padding[1] + param.padding[3] - dilated_ksize_x) / param.strides[1] + 1;
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, ConvertLayout(oshape, kNCHW, out_layout));
  // Perform incomplete shape inference. Fill in the missing values in data shape.
  // 1) We can always fill in the batch_size.
  // 2) We can back-calculate the input height/width if the corresponding stride is 1.
  oshape = ConvertLayout((*out_shape)[0], out_layout, kNCHW);
  dshape[0] = oshape[0];
  if (oshape[2] && param.strides[0] == 1) {
    dshape[2] = oshape[2] + dilated_ksize_y - 1 - param.padding[0] + param.padding[2];
  }
  if (oshape[3] && param.strides[1] == 1) {
    dshape[3] = oshape[3] + dilated_ksize_x - 1 - param.padding[1] + param.padding[3];
  }
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, Conv2DParam::kData,
                          ConvertLayout(dshape, kNCHW, in_layout));
  // Check whether the kernel sizes are valid
  if (dshape[2] != 0) {
    CHECK_LE(dilated_ksize_y, dshape[2] + 2 * param.padding[0])
      << "kernel size exceed input";
  }
  if (dshape[3] != 0) {
    CHECK_LE(dilated_ksize_x, dshape[3] + 2 * param.padding[1])
        << "kernel size exceed input";
  }
  return true;
}

DMLC_REGISTER_PARAMETER(BitserialConv2DParam);
template <typename PARAM>
inline bool BitserialConv2DInferType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_type,
                            std::vector<int>* out_type) {
  const PARAM& param = nnvm::get<PARAM>(attrs.parsed);
  // if (param.use_bias) {
  //   CHECK_EQ(in_type->size(), 3U) << "Input:[data, weight, bias]";
  // } else {
  //   CHECK_EQ(in_type->size(), 2U) << "Input:[data, weight]";
  // }
  CHECK_EQ(out_type->size(), 1U);
  if (param.out_dtype != -1) { // This is probably wrong - fix later
    int weight = param.bit_axis > -1 ? param.pack_dtype : param.out_dtype;

    CHECK(!type_is_none((*in_type)[0]));
    for (size_t i = 2; i < in_type->size(); ++i) {
      NNVM_ASSIGN_INPUT_TYPE(attrs, *in_type, i, param.out_dtype);
    }

    int out = param.pack_outputs ? param.pack_dtype : param.out_dtype;
    NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_type, 0, out);
  } else {
    ElemwiseType<-1, 1>(attrs, in_type, out_type);
  }
  return true;
}

inline bool BitserialConv2DInferShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_shape,
                             std::vector<TShape>* out_shape) {
  const BitserialConv2DParam& param = nnvm::get<BitserialConv2DParam>(attrs.parsed);

  const Layout in_layout(param.layout);
  const Layout kernel_layout(param.kernel_layout);
  Layout out_layout(param.out_layout);
  if (!out_layout.defined()) out_layout = in_layout;

  // if (param.pack_outputs) {
  //   CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  // } else {
  //   CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  // }
  CHECK_EQ(out_shape->size(), 1U);

  TShape dshape = in_shape->at(0);
  if (dshape.ndim() == 0) return false;

  // CHECK_EQ(dshape.ndim(), 4U) << "Input data should be 4D";
  CHECK_EQ(param.kernel_size.ndim(), 2U);
  CHECK_EQ(param.strides.ndim(), 2U)
      << "incorrect stride size: " << param.strides;

  // Weight shape - check to see if the bit_axis is filled in, then the weights are prepacked
  std::vector<dim_t> shape{param.kernel_size[0],
                             param.kernel_size[1],
                             dshape[3],
                             param.channels};
  const int bit_axis(param.bit_axis);
  auto shape_itr = shape.begin();
  if (bit_axis != -1) {
    dim_t pack_amount = 8;
    if (param.pack_dtype == 8)
      pack_amount = 16;
    else if (param.pack_dtype == 9)
      pack_amount = 32;
    else if (param.pack_dtype == 10)
      pack_amount = 64;

    shape[2] = shape[2] / pack_amount;
    shape.insert(shape_itr + bit_axis, 1, param.weight_bits);
  } 
  TShape wshape(shape);
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, BitserialConv2DParam::kWeight, wshape);

  // If using a fused pack, then need to add rounding bias
  if (param.pack_outputs) {
    static const Layout default_bias_layout("C");
    TShape bias_shape({param.channels});
    auto oc_block = out_layout.subsizeof('C');
    if (oc_block > 0) {
      size_t split_axis = (out_layout.indexof('C') < out_layout.indexof('c')) ? 1 : 0;
      bias_shape = ConvertLayout(bias_shape, default_bias_layout,
                                 default_bias_layout.split('C', split_axis, oc_block));
    }
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, BitserialConv2DParam::kRound, bias_shape);
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, BitserialConv2DParam::kClipMin, bias_shape);
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, BitserialConv2DParam::kClipMax, bias_shape);
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, BitserialConv2DParam::kRShift, bias_shape);
    TShape oshape({dshape[0], 0, 0, param.activation_bits, param.channels / 8});
    if (dshape[1] != 0) {
      oshape[1] = (dshape[1] - param.kernel_size[0] + param.padding[0] + param.padding[2]) / param.strides[0] + 1;
    }
    if (dshape[2] != 0) {
      oshape[2] = (dshape[2] - param.kernel_size[1] + param.padding[1] + param.padding[3]) / param.strides[1] + 1;
    }
    if (param.use_pool) {
      dim_t pad_h, pad_w;
      if (param.pool_padding.ndim() == 1) {
        pad_h = param.pool_padding[0] * 2;
        pad_w = param.pool_padding[0] * 2;
      } else if (param.pool_padding.ndim() == 2) {
        // (top, left)
        pad_h = param.pool_padding[0] * 2;
        pad_w = param.pool_padding[1] * 2;
      } else if (param.pool_padding.ndim() == 4) {
        // (top, left, bottom, right)
        pad_h = param.pool_padding[0] + param.pool_padding[2];
        pad_w = param.pool_padding[1] + param.pool_padding[3];
      }
      oshape[1] = ((oshape[1] + pad_h - param.pool_size[0] +
                  param.pool_strides[0] - 1) / param.pool_strides[0]) + 1;
      oshape[2] = ((oshape[2] + pad_w - param.pool_size[1] +
                  param.pool_strides[1] - 1) / param.pool_strides[1]) + 1;
    }
    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  } else {
    // Normal bitserial_convd2d with no output packing
    // Assumes NHWC
    TShape oshape({dshape[0], 0, 0, param.channels});
    if (dshape[1] != 0) {
      oshape[1] = (dshape[1] - param.kernel_size[0] + param.padding[0] + param.padding[2]) / param.strides[0] + 1;
    }
    if (dshape[2] != 0) {
      oshape[2] = (dshape[2] - param.kernel_size[1] + param.padding[1] + param.padding[3]) / param.strides[1] + 1;
    }

    if (param.use_pool) {
      dim_t pad_h, pad_w;
      if (param.pool_padding.ndim() == 1) {
        pad_h = param.pool_padding[0] * 2;
        pad_w = param.pool_padding[0] * 2;
      } else if (param.pool_padding.ndim() == 2) {
        // (top, left)
        pad_h = param.pool_padding[0] * 2;
        pad_w = param.pool_padding[1] * 2;
      } else if (param.pool_padding.ndim() == 4) {
        // (top, left, bottom, right)
        pad_h = param.pool_padding[0] + param.pool_padding[2];
        pad_w = param.pool_padding[1] + param.pool_padding[3];
      }
      oshape[1] = ((oshape[1] + pad_h - param.pool_size[0] +
                  param.pool_strides[0] - 1) / param.pool_strides[0]) + 1;
      oshape[2] = ((oshape[2] + pad_w - param.pool_size[1] +
                  param.pool_strides[1] - 1) / param.pool_strides[1]) + 1;
    }
    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  }
  
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, Conv2DParam::kData, dshape);
  return true;
}

template<typename PARAM>
inline bool BitserialConv2DCorrectLayout(const NodeAttrs& attrs,
                                std::vector<Layout> *ilayouts,
                                const std::vector<Layout> *last_ilayouts,
                                std::vector<Layout> *olayouts) {
  const PARAM& param = nnvm::get<PARAM>(attrs.parsed);

  const Layout in_layout(param.layout);
  Layout out_layout(param.out_layout);
  if (!out_layout.defined()) out_layout = in_layout;

  const Layout kernel_layout(param.kernel_layout);
  if (param.pack_outputs) {
    CHECK_EQ(ilayouts->size(), 6U) << "Input:[data, weight, round, clip_a_min, clip_a_max, rshift]";
    NNVM_ASSIGN_LAYOUT(*ilayouts, 0, in_layout);
    NNVM_ASSIGN_LAYOUT(*ilayouts, 1, kernel_layout);
    // automatically decide bias layout
    Layout bias_layout("C");
    auto oc_block = out_layout.subsizeof('C');
    if (oc_block > 0) {
      size_t split_axis = (out_layout.indexof('C') < out_layout.indexof('c')) ? 1 : 0;
      bias_layout = bias_layout.split('C', split_axis, oc_block);
    }
    NNVM_ASSIGN_LAYOUT(*ilayouts, 2, bias_layout);
    NNVM_ASSIGN_LAYOUT(*ilayouts, 3, bias_layout);
    NNVM_ASSIGN_LAYOUT(*ilayouts, 4, bias_layout);
    NNVM_ASSIGN_LAYOUT(*ilayouts, 5, bias_layout);
  } else {
    // CHECK_EQ(ilayouts->size(), 2U) << "Input:[data, weight]";
    NNVM_ASSIGN_LAYOUT(*ilayouts, 0, in_layout);
    NNVM_ASSIGN_LAYOUT(*ilayouts, 1, kernel_layout);
  }

  CHECK_EQ(olayouts->size(), 1U);
  NNVM_ASSIGN_LAYOUT(*olayouts, 0, out_layout);

  return true;
}

inline bool WinogradConv2DInferShape(const nnvm::NodeAttrs& attrs,
                                     std::vector<TShape>* in_shape,
                                     std::vector<TShape>* out_shape) {
  static const Layout kNCHW("NCHW");
  static const Layout kOIHW("OIHW");

  const WinogradConv2DParam& param = nnvm::get<WinogradConv2DParam>(attrs.parsed);

  const Layout in_layout(param.layout);
  const Layout kernel_layout(param.kernel_layout);
  CHECK(in_layout.convertible(kNCHW))
    << "Conv only support input layouts that are convertible from NCHW."
    << " But got " << in_layout;
  CHECK(kernel_layout.convertible(kOIHW))
    << "Conv only support kernel layouts that are convertible from OIHW."
    << " But got "<< kernel_layout;

  Layout out_layout(param.out_layout);
  if (!out_layout.defined()) out_layout = in_layout;
  CHECK(out_layout.convertible(kNCHW))
    << "Conv only support output layouts that are convertible from NCHW."
    << " But got " << out_layout;

  if (param.use_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_shape->size(), 1U);

  TShape dshape = in_shape->at(0);
  if (dshape.ndim() == 0) return false;
  dshape = ConvertLayout(dshape, in_layout, kNCHW);

  CHECK_EQ(dshape.ndim(), 4U) << "Input data should be 4D";
  CHECK_EQ(param.kernel_size.ndim(), 2U);
  CHECK_EQ(param.strides.ndim(), 2U)
      << "incorrect stride size: " << param.strides;
  CHECK_EQ(param.dilation.ndim(), 2U)
      << "incorrect dilate size: " << param.dilation;
  CHECK_EQ(dshape[1] % param.groups, 0U)
      << "input channels must divide group size";
  CHECK_EQ(param.channels % param.groups, 0U)
      << "output channels must divide group size";

  // NOTE: Do not check weight shape here!
  // Different backend requires different layout to compute
  // the batch gemm stage in winograd efficiently, but we want to
  // make this NNVM symbol work for all backends.
  // So we accept all weight shapes, and assume the TOPI developers
  // can handle this correctly in alter_op_layout.

  if (param.use_bias) {
    static const Layout default_bias_layout("C");
    TShape bias_shape({param.channels});
    auto oc_block = out_layout.subsizeof('C');
    if (oc_block > 0) {
      size_t split_axis = (out_layout.indexof('C') < out_layout.indexof('c')) ? 1 : 0;
      bias_shape = ConvertLayout(bias_shape, default_bias_layout,
                                 default_bias_layout.split('C', split_axis, oc_block));
    }
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, WinogradConv2DParam::kBias, bias_shape);
  }
  // dilation
  dim_t dilated_ksize_y = 1 + (param.kernel_size[0] - 1) * param.dilation[0];
  dim_t dilated_ksize_x = 1 + (param.kernel_size[1] - 1) * param.dilation[1];
  TShape oshape({dshape[0], param.channels, 0, 0});
  if (dshape[2] != 0) {
    oshape[2] = (dshape[2] + param.padding[0] * 2 - dilated_ksize_y) / param.strides[0] + 1;
  }
  if (dshape[3] != 0) {
    oshape[3] = (dshape[3] + param.padding[1] * 2 - dilated_ksize_x) / param.strides[1] + 1;
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, ConvertLayout(oshape, kNCHW, out_layout));
  // Perform incomplete shape inference. Fill in the missing values in data shape.
  // 1) We can always fill in the batch_size.
  // 2) We can back-calculate the input height/width if the corresponding stride is 1.
  oshape = ConvertLayout((*out_shape)[0], out_layout, kNCHW);
  dshape[0] = oshape[0];
  if (oshape[2] && param.strides[0] == 1) {
    dshape[2] = oshape[2] + dilated_ksize_y - 1 - 2 * param.padding[0];
  }
  if (oshape[3] && param.strides[1] == 1) {
    dshape[3] = oshape[3] + dilated_ksize_x - 1 - 2 * param.padding[1];
  }
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, WinogradConv2DParam::kData,
                          ConvertLayout(dshape, kNCHW, in_layout));
  // Check whether the kernel sizes are valid
  if (dshape[2] != 0) {
    CHECK_LE(dilated_ksize_y, dshape[2] + 2 * param.padding[0])
      << "kernel size exceed input";
  }
  if (dshape[3] != 0) {
    CHECK_LE(dilated_ksize_x, dshape[3] + 2 * param.padding[1])
      << "kernel size exceed input";
  }
  return true;
}

template <typename PARAM>
inline bool Conv2DInferType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_type,
                            std::vector<int>* out_type) {
  const PARAM& param = nnvm::get<PARAM>(attrs.parsed);
  if (param.use_bias) {
    CHECK_EQ(in_type->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_type->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_type->size(), 1U);
  if (param.out_dtype != -1) {
    CHECK(!type_is_none((*in_type)[0]));
    for (size_t i = 1; i < in_type->size(); ++i) {
      NNVM_ASSIGN_INPUT_TYPE(attrs, *in_type, i, (*in_type)[i]);
    }
    NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_type, 0, param.out_dtype);
  } else {
    ElemwiseType<-1, 1>(attrs, in_type, out_type);
  }
  return true;
}


template<typename PARAM>
inline bool Conv2DCorrectLayout(const NodeAttrs& attrs,
                                std::vector<Layout> *ilayouts,
                                const std::vector<Layout> *last_ilayouts,
                                std::vector<Layout> *olayouts) {
  const PARAM& param = nnvm::get<PARAM>(attrs.parsed);

  const Layout in_layout(param.layout);
  Layout out_layout(param.out_layout);
  if (!out_layout.defined()) out_layout = in_layout;

  const Layout kernel_layout(param.kernel_layout);
  if (param.use_bias) {
    CHECK_EQ(ilayouts->size(), 3U) << "Input:[data, weight, bias]";
    NNVM_ASSIGN_LAYOUT(*ilayouts, 0, in_layout);
    NNVM_ASSIGN_LAYOUT(*ilayouts, 1, kernel_layout);
    // automatically decide bias layout
    Layout bias_layout("C");
    auto oc_block = out_layout.subsizeof('C');
    if (oc_block > 0) {
      size_t split_axis = (out_layout.indexof('C') < out_layout.indexof('c')) ? 1 : 0;
      bias_layout = bias_layout.split('C', split_axis, oc_block);
    }
    NNVM_ASSIGN_LAYOUT(*ilayouts, 2, bias_layout);
  } else {
    CHECK_EQ(ilayouts->size(), 2U) << "Input:[data, weight]";
    NNVM_ASSIGN_LAYOUT(*ilayouts, 0, in_layout);
    NNVM_ASSIGN_LAYOUT(*ilayouts, 1, kernel_layout);
  }

  CHECK_EQ(olayouts->size(), 1U);
  NNVM_ASSIGN_LAYOUT(*olayouts, 0, out_layout);

  return true;
}

NNVM_REGISTER_OP(conv2d)
.describe(R"code(2D convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of
outputs. If `use_bias` is True,
a bias vector is created and added to the outputs.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
- **bias**: (channels,)
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_argument("weight", "4D Tensor", "Weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(Conv2DParam::__FIELDS__())
.set_attr_parser(ParamParser<Conv2DParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<Conv2DParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<Conv2DParam>)
.set_attr<FInferShape>("FInferShape", Conv2DInferShape)
.set_attr<FInferType>("FInferType", Conv2DInferType<Conv2DParam>)
.set_attr<FCorrectLayout>("FCorrectLayout", Conv2DCorrectLayout<Conv2DParam>)
.set_num_outputs(1)
.set_num_inputs(UseBiasNumInputs<Conv2DParam>)
.set_support_level(2)
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    return MakeGradNode("_conv2d_grad", n,
                        {ograds[0], n->inputs[Conv2DParam::kData],
                         n->inputs[Conv2DParam::kWeight]},
                        n->attrs.dict);
});

NNVM_REGISTER_OP(_contrib_conv2d_NCHWc)
.describe(R"code(2D convolution layer (e.g. spatial convolution over images).
)code" NNVM_ADD_FILELINE)
.add_argument("data", "5D Tensor", "Packed input data.")
.add_argument("weight", "6D Tensor", "Packed weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(Conv2DParam::__FIELDS__())
.set_attr_parser(ParamParser<Conv2DParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<Conv2DParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<Conv2DParam>)
.set_attr<FInferShape>("FInferShape", Conv2DInferShape)
.set_attr<FInferType>("FInferType", Conv2DInferType<Conv2DParam>)
.set_attr<FCorrectLayout>("FCorrectLayout", Conv2DCorrectLayout<Conv2DParam>)
.set_num_outputs(1)
.set_num_inputs(UseBiasNumInputs<Conv2DParam>)
.set_support_level(2);

NNVM_REGISTER_OP(_contrib_conv2d_winograd_weight_transform)
.describe(R"code(Weight transformation of winograd fast convolution algorithm.
Separate this into another nnvm symbol in order to enable Precompute Pass to compute the
weight transformation in advance.

- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
)code" NNVM_ADD_FILELINE)
.add_argument("weight", "4D Tensor", "Weight tensor.")
.add_arguments(WinogradWeightTransformParam::__FIELDS__())
.set_attr_parser(ParamParser<WinogradWeightTransformParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<WinogradWeightTransformParam>)
.set_attr<FInferShape>("FInferShape", [](const nnvm::NodeAttrs& attrs,
                                         std::vector<TShape> *in_shape,
                                         std::vector<TShape> *out_shape) {
  const auto& param = nnvm::get<WinogradWeightTransformParam>(attrs.parsed);
  const TShape &wshape = (*in_shape)[0];

  CHECK_EQ(wshape.ndim(), 4) << "Weight should be a 4 dimensional tensor";

  TShape oshape({param.tile_size + wshape[2] - 1,
                 param.tile_size + wshape[3] - 1,
                 wshape[0],
                 wshape[1]});
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
  })
.set_attr<FCorrectLayout>("FCorrectLayot", [](const NodeAttrs& attrs,
                                              std::vector<Layout> *ilayouts,
                                              const std::vector<Layout> *last_ilayouts,
                                              std::vector<Layout> *olayouts) {
  Layout layout("OIHW");
  NNVM_ASSIGN_LAYOUT(*ilayouts, 0, layout);
  NNVM_ASSIGN_LAYOUT(*olayouts, 0, layout);
  return true;
})
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_num_outputs(1)
.set_num_inputs(1)
.set_support_level(5);

DMLC_REGISTER_PARAMETER(WinogradWeightTransformParam);

NNVM_REGISTER_OP(_contrib_conv2d_winograd_without_weight_transform)
.describe(R"code(Compute conv2d with winograd algorithm.

- **data**: Input is 4D array of shape  (batch_size, in_channels, height, width)
- **weight**: Any shape
            We do not check shape for this input tensor.

- **bias**: (channels,)
- **out**:  Output is 4D array of shape (batch_size, channels, out_height, out_width)
)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_argument("weight", "Tensor", "Transformed weight tensor.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(WinogradConv2DParam::__FIELDS__())
.set_attr_parser(ParamParser<WinogradConv2DParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<WinogradConv2DParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<WinogradConv2DParam>)
.set_attr<FInferShape>("FInferShape", WinogradConv2DInferShape)
.set_attr<FInferType>("FInferType", Conv2DInferType<WinogradConv2DParam>)
.set_attr<FCorrectLayout>("FCorrectLayout", Conv2DCorrectLayout<WinogradConv2DParam>)
.set_num_outputs(1)
.set_num_inputs(UseBiasNumInputs<WinogradConv2DParam>)
.set_support_level(5);

DMLC_REGISTER_PARAMETER(WinogradConv2DParam);

NNVM_REGISTER_OP(_conv2d_grad)
  .describe(R"code(2D convolution grad.

)code" NNVM_ADD_FILELINE)
.add_argument("ograd", "4D Tensor", "Output grad.")
.add_argument("data", "4D Tensor", "Input data of conv2d.")
.add_argument("weight", "4D Tensor", "Input weight.")
.set_num_inputs(3)
.set_num_outputs(UseBiasNumInputs<Conv2DParam>)
.set_attr<FListOutputNames>("FListOutputNames", UseBiasListInputNames<Conv2DParam>)
.set_attr_parser(ParamParser<Conv2DParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<Conv2DParam>)
.set_attr<FInferShape>(
  "FInferShape", [](const nnvm::NodeAttrs& attrs,
                    std::vector<TShape>* in_attrs,
                    std::vector<TShape>* out_attrs) {
    const Conv2DParam& param = nnvm::get<Conv2DParam>(attrs.parsed);
    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, Conv2DParam::kData, in_attrs->at(1));
    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, Conv2DParam::kWeight, in_attrs->at(2));
    if (param.use_bias) {
      NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, Conv2DParam::kBias, TShape({param.channels}));
    }
    return true;
})
.set_attr<FInferType>("FInferType", ElemwiseType<3, -1>)
.set_attr<TIsBackward>("TIsBackward", true);


DMLC_REGISTER_PARAMETER(Conv2DTransposeParam);

inline bool Conv2DTransposeInferShape(const nnvm::NodeAttrs& attrs,
                                      std::vector<TShape>* in_shape,
                                      std::vector<TShape>* out_shape) {
  static const Layout kNCHW("NCHW");
  static const Layout kOIHW("OIHW");
  const Conv2DTransposeParam& param = nnvm::get<Conv2DTransposeParam>(attrs.parsed);
  const Layout layout(param.layout);
  const Layout kernel_layout(param.kernel_layout);
  if (param.use_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_shape->size(), 1U);

  const TShape& dshape = (*in_shape)[Conv2DTransposeParam::kData];
  if (dshape.ndim() ==  0) return false;
  TShape dshape_nchw = ConvertLayout(dshape, layout, kNCHW);

  CHECK_EQ(dshape_nchw[1] % param.groups, 0U)
      << "input num_filter must divide group size";
  CHECK_EQ(param.channels % param.groups, 0U)
      << "output num_filter must divide group size";
  CHECK_EQ(param.kernel_size.ndim(), 2U)
      << "incorrect kernel size: " << param.kernel_size;
  CHECK_EQ(param.strides.ndim(), 2U)
      << "incorrect stride size: " << param.strides;
  CHECK_EQ(param.dilation.ndim(), 2U)
      << "incorrect dilate size: " << param.dilation;

  TShape wshape({dshape_nchw[1],
                 param.channels / param.groups,
                 param.kernel_size[0],
                 param.kernel_size[1]});
  wshape = ConvertLayout(wshape, kOIHW, kernel_layout);
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, Conv2DTransposeParam::kWeight, wshape);

  if (param.use_bias) {
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape,
                            Conv2DTransposeParam::kBias,
                            TShape({param.channels}));
  }
  // dilation
  dim_t dilated_ksize_y = 1 + (param.kernel_size[0] - 1) * param.dilation[0];
  dim_t dilated_ksize_x = 1 + (param.kernel_size[1] - 1) * param.dilation[1];
  // output shape.
  TShape oshape({dshape_nchw[0], param.channels, 0, 0});
  oshape[2] = (param.strides[0] * (dshape_nchw[2] - 1) + dilated_ksize_y -
               2 * param.padding[0] + param.output_padding[0]);

  oshape[3] = (param.strides[1] * (dshape_nchw[3] - 1) + dilated_ksize_x -
               2 * param.padding[1] + param.output_padding[1]);
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0,
                           ConvertLayout(oshape, kNCHW, layout));
  return true;
}

inline bool Conv2DTransposeCorrectLayout(const NodeAttrs& attrs,
                                         std::vector<Layout> *ilayouts,
                                         const std::vector<Layout> *last_ilayouts,
                                         std::vector<Layout> *olayouts) {
  const Conv2DTransposeParam& param = nnvm::get<Conv2DTransposeParam>(attrs.parsed);

  const Layout in_layout(param.layout);

  const Layout kernel_layout(param.kernel_layout);
  if (param.use_bias) {
    CHECK_EQ(ilayouts->size(), 3U) << "Input:[data, weight, bias]";
    NNVM_ASSIGN_LAYOUT(*ilayouts, 0, in_layout);
    NNVM_ASSIGN_LAYOUT(*ilayouts, 1, kernel_layout);
    NNVM_ASSIGN_LAYOUT(*ilayouts, 2, Layout("C"));
  } else {
    CHECK_EQ(ilayouts->size(), 2U) << "Input:[data, weight]";
    NNVM_ASSIGN_LAYOUT(*ilayouts, 0, in_layout);
    NNVM_ASSIGN_LAYOUT(*ilayouts, 1, kernel_layout);
  }

  CHECK_EQ(olayouts->size(), 1U);
  NNVM_ASSIGN_LAYOUT(*olayouts, 0, in_layout);

  return true;
}

NNVM_REGISTER_OP(conv2d_transpose)
.describe(R"code(Transposed 2D convolution layer (sometimes called Deconvolution).

The need for transposed convolutions generally arises
from the desire to use a transformation going in the opposite direction
of a normal convolution, i.e., from something that has the shape of the
output of some convolution to something that has the shape of its input
while maintaining a connectivity pattern that is compatible with
said convolution.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (in_channels, channels, kernel_size[0], kernel_size[1])
- **bias**: (channels,)
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
v            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

            out_height and out_width are calculated as::
                out_height = (height-1)*strides[0]-2*padding[0]+kernel_size[0]+output_padding[0]
                out_width = (width-1)*strides[1]-2*padding[1]+kernel_size[1]+output_padding[1]

)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_argument("weight", "4D Tensor", "Weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(Conv2DTransposeParam::__FIELDS__())
.set_attr_parser(ParamParser<Conv2DTransposeParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<Conv2DTransposeParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<Conv2DTransposeParam>)
.set_attr<FInferShape>("FInferShape", Conv2DTransposeInferShape)
.set_attr<FInferType>("FInferType", Conv2DInferType<Conv2DTransposeParam>)
.set_attr<FCorrectLayout>("FCorrectLayout", Conv2DTransposeCorrectLayout)
.set_num_outputs(1)
.set_num_inputs(UseBiasNumInputs<Conv2DTransposeParam>)
.set_support_level(2);

// Note: Relying on many of Conv2d's helper functions
NNVM_REGISTER_OP(bitserial_conv2d)
.describe(R"code(2D bitserial convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of
outputs. If `use_bias` is True,
a bias vector is created and added to the outputs.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
- **bias**: (channels,)
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_argument("weight", "4D Tensor", "Weight matrix.")
.add_argument("round", "1D Tensor", "Bias parameter.")
.add_argument("clip_a_min", "1D Tensor", "Clip minimum.")
.add_argument("clip_a_max", "1D Tensor", "Clip maximum.")
.add_argument("rshift", "1D Tensor", "Right shift.")
.add_arguments(BitserialConv2DParam::__FIELDS__())
.set_attr_parser(ParamParser<BitserialConv2DParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<BitserialConv2DParam>)
.set_attr<FListInputNames>("FListInputNames", BitserialListInputNames<BitserialConv2DParam>)
.set_attr<FInferShape>("FInferShape", BitserialConv2DInferShape)
.set_attr<FInferType>("FInferType", BitserialConv2DInferType<BitserialConv2DParam>)
.set_attr<FCorrectLayout>("FCorrectLayout", BitserialConv2DCorrectLayout<BitserialConv2DParam>)
.set_num_outputs(1)
.set_num_inputs(6)
.set_support_level(2); // TODO: not certain about this support level


}  // namespace top
}  // namespace nnvm
