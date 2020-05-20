#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("NeuropodAddition")
    .Input("x: float")
    .Input("y: float")
    .Output("out: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        // TODO: check that x and y have the same shape
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class AdditionOp : public OpKernel
{
public:
    explicit AdditionOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensors
        const Tensor &x      = context->input(0);
        const Tensor &y      = context->input(1);
        auto          x_flat = x.flat<float>();
        auto          y_flat = y.flat<float>();

        // Create an output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, x.shape(), &output_tensor));
        auto output_flat = output_tensor->flat<float>();

        // Add the tensors
        const int N = output_flat.size();
        for (int i = 0; i < N; i++)
        {
            output_flat(i) = x_flat(i) + y_flat(i);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("NeuropodAddition").Device(DEVICE_CPU), AdditionOp);
