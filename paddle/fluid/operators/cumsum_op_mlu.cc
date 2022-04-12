/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class CumSumMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    int axis = ctx.Attr<int>("axis");
    bool exclusive = ctx.Attr<bool>("exclusive");
    bool reverse = ctx.Attr<bool>("reverse");

    out->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc out_desc(*out);

    bool flatten = ctx.Attr<bool>("flatten");
    if (flatten) {
      PADDLE_ENFORCE_EQ(
          axis, -1,
          platform::errors::InvalidArgument(
              "when flatten is true, attr axis must be default %d, but got %d",
              -1, axis));

      Tensor flat_x(x->type());
      flat_x.ShareDataWith(*x);
      flat_x.Resize(phi::make_ddim({x->numel()}));

      MLUCnnlTensorDesc x_desc(flat_x);
      MLUCnnl::Cumsum(ctx, axis, exclusive, reverse, x_desc.get(),
                      GetBasePtr(&flat_x), out_desc.get(), GetBasePtr(out));
    } else {
      MLUCnnlTensorDesc x_desc(x);
      MLUCnnl::Cumsum(ctx, axis, exclusive, reverse, x_desc.get(),
                      GetBasePtr(x), out_desc.get(), GetBasePtr(out));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(cumsum, ops::CumSumMLUKernel<int>,
                       ops::CumSumMLUKernel<float>,
                       ops::CumSumMLUKernel<plat::float16>);
