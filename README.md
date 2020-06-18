# Neuropod

## What is Neuropod?

[Neuropod](https://github.com/uber/neuropod) is a library that provides a uniform interface to run deep learning models from multiple frameworks in C++ and Python. Neuropod makes it easy for researchers to build models in a framework of their choosing while also simplifying productionization of these models.

It currently supports TensorFlow, PyTorch, TorchScript, Keras and [Ludwig](http://ludwig.ai).

## Why use Neuropod?

#### Run models from any supported framework using one API

Running a TensorFlow model looks exactly like running a PyTorch model.

```py
x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])

for model_path in [TF_ADDITION_MODEL_PATH, PYTORCH_ADDITION_MODEL_PATH]:
    # Load the model
    neuropod = load_neuropod(model_path)

    # Run inference
    results = neuropod.infer({"x": x, "y": y})

    # array([6, 8, 10, 12])
    print results["out"]
```

See the [tutorial](https://neuropod.ai/tutorial/), [Python guide](https://neuropod.ai/pyguide/), or [C++ guide](https://neuropod.ai/cppguide/) for more examples.

Some benefits of this include:

- All of your inference code is framework agnostic.
- You can easily switch between deep learning frameworks if necessary without changing runtime code.
- Avoid the learning curve of using the C++ libtorch API and the C/C++ TF API

Any Neuropod model can be run from both C++ and Python (even PyTorch models that have not been converted to TorchScript).

#### Define a Problem API

This lets you focus more on the problem you're solving rather than the framework you're using to solve it.

For example, if you define a problem API for 2d object detection, any model that implements it can reuse all the existing inference code and infrastructure for that problem.

```py
INPUT_SPEC = [
    # BGR image
    {"name": "image", "dtype": "uint8", "shape": (1200, 1920, 3)},
]

OUTPUT_SPEC = [
    # shape: (num_detections, 4): (xmin, ymin, xmax, ymax)
    # These values are in units of pixels. The origin is the top left corner
    # with positive X to the right and positive Y towards the bottom of the image
    {"name": "boxes", "dtype": "float32", "shape": ("num_detections", 4)},

    # The list of classes that the network can output
    # This must be some subset of ['vehicle', 'person', 'motorcycle', 'bicycle']
    {"name": "supported_object_classes", "dtype": "string", "shape": ("num_classes",)},

    # The probability of each class for each detection
    # These should all be floats between 0 and 1
    {"name": "object_class_probability", "dtype": "float32", "shape": ("num_detections", "num_classes")},
]
```

This lets you

- Build a single metrics pipeline for a problem
- Easily compare models solving the same problem (even if they're in different frameworks)
- Build optimized inference code that can run any model that solves a particular problem
- Swap out models that solve the same problem at runtime with no code change (even if the models are from different frameworks)
- Run fast experiments

See the [tutorial](https://neuropod.ai/tutorial/) for more details.

#### Build generic tools and pipelines

If you have several models that take in a similar set of inputs, you can build and optimize one framework-agnostic input generation pipeline and share it across models.

#### Other benefits

- Fully self-contained models (including custom ops)
- [Efficient zero-copy operations](https://neuropod.ai/advanced/efficient_tensor_creation/)
- [Tested on](https://neuropod.ai/developing/#build-matrix) platforms including
    - Mac, Linux, Linux (GPU)
    - Four or five versions of each supported framework
    - Five versions of Python

- Model isolation with [out-of-process execution](https://neuropod.ai/advanced/ope/)
    - Use multiple different versions of frameworks in the same application
        - Ex: Experimental models using Torch nightly along with models using Torch 1.1.0
- Switch from running in-process to running out-of-process with [one line of code](https://neuropod.ai/advanced/ope/)

## Getting started

See the [basic introduction tutorial](https://neuropod.ai/tutorial/) for an overview of how to get started with Neuropod.

The [Python guide](https://neuropod.ai/pyguide/) and [C++ guide](https://neuropod.ai/cppguide/) go into more detail on running Neuropod models.
