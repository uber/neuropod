# Neuropods Tutorial

In this tutorial, we’re going to build a simple Neuropod model for addition in TensorFlow, PyTorch, and TorchScript. We'll also show how to run inference from Python and C++.

Almost all of the examples/code in this tutorial come from the Neuropods unit and integration tests. Please read through them for complete working examples.

The Neuropod packaging and inference interfaces also have comprehensive docstrings and provide a more detailed usage of the API than this tutorial.

## Package a Model

The first step for packaging a model is to define a “problem” (e.g. 2d object detection).

A “problem” is composed of 4 things:
- an `input_spec`
  - A list of dicts specifying the name, datatype, and shape of an input tensor
- an `output_spec`
  - A list of dicts specifying the name, datatype, and shape of an output tensor
- `test_input_data` (optional)
  - If provided, Neuropods will run inference immediately after packaging to verify that the model was packaged correctly. Must be provided if `test_output_data` is provided
- `test_output_data` (optional)
  - If provided, Neuropods will test that the output of inference with `test_input_data` matches `test_output_data`

The shape of a tensor can include `None` in which case any value is acceptable. You can also use “symbols” in these shape definitions. Every instance of that symbol must resolve to the same value at runtime.

For example, here’s a problem definition for our addition model:

```py
INPUT_SPEC = [
    # A one dimensional tensor of any size with dtype float32
    {"name": "x", "dtype": "float32", "shape": ("num_inputs",)},
    # A one dimensional tensor of the same size with dtype float32
    {"name": "y", "dtype": "float32", "shape": ("num_inputs",)},
]

OUTPUT_SPEC = [
    # The sum of the two tensors
    {"name": "out", "dtype": "float32", "shape": (None,)},
]

TEST_INPUT_DATA = {
    "x": np.arange(5, dtype=np.float32),
    "y": np.arange(5, dtype=np.float32),
}

TEST_EXPECTED_OUT = {
    "out": np.arange(5) + np.arange(5)
}
```

The symbol `num_inputs` in the shapes of `x` and `y` must resolve to the same value at runtime.

For a definition of a “real” problem, see the example problem definitions section in the appendix.

Now that we have a problem defined, we’re going to see how to package a model in each of the three currently supported DL frameworks.


### TensorFlow

There are two ways to package a TensorFlow model. One is with a `GraphDef` the other is with a path to a frozen graph. Both of these require a `node_name_mapping` that maps a tensor name in the problem definition (see above) to a node in the TensorFlow graph. See the examples below for more detail.

#### GraphDef

```py
def create_tf_addition_model():
    """
    A simple addition model
    """
    g = tf.Graph()
    with g.as_default():
        with tf.name_scope("some_namespace"):
            x = tf.placeholder(tf.float32, name="in_x")
            y = tf.placeholder(tf.float32, name="in_y")

            out = tf.add(x, y, name="out")

    return g.as_graph_def()

# `create_tensorflow_neuropod` runs inference with the test data immediately
# after creating the neuropod. Raises a ValueError if the model output
# does not match the expected output.
create_tensorflow_neuropod(
    neuropod_path=neuropod_path,
    model_name="addition_model",
    graph_def=create_tf_addition_model(),
    node_name_mapping={
        "x": "some_namespace/in_x:0",
        "y": "some_namespace/in_y:0",
        "out": "some_namespace/out:0",
    },
    input_spec=addition_problem_definition.INPUT_SPEC,
    output_spec=addition_problem_definition.OUTPUT_SPEC,
    test_input_data=addition_problem_definition.TEST_INPUT_DATA,
    test_expected_out=addition_problem_definition.TEST_EXPECTED_OUT,
)
```

#### Path to a Frozen Graph

If you already have a frozen graph, you can package the model like this:

```py
# `create_tensorflow_neuropod` runs inference with the test data immediately
# after creating the neuropod. Raises a ValueError if the model output
# does not match the expected output.
create_tensorflow_neuropod(
    neuropod_path=neuropod_path,
    model_name="addition_model",
    frozen_graph_path="/path/to/my/frozen.graph",
    node_name_mapping={
        "x": "some_namespace/in_x:0",
        "y": "some_namespace/in_y:0",
        "out": "some_namespace/out:0",
    },
    input_spec=addition_problem_definition.INPUT_SPEC,
    output_spec=addition_problem_definition.OUTPUT_SPEC,
    test_input_data=addition_problem_definition.TEST_INPUT_DATA,
    test_expected_out=addition_problem_definition.TEST_EXPECTED_OUT,
)
```

### PyTorch

Packaging a PyTorch model is a bit more complicated because you need python code and the weights in order to run the network.

Let's say our addition model looks like this (and is stored at `/my/model/code/dir/main.py`):

```py
import torch
import torch.nn as nn

class AdditionModel(nn.Module):
  def forward(self, x, y):
      return {
          "out": x + y
      }

def get_model(data_root):
  return AdditionModel()
```

In order to package it, we need 4 things:
- The paths to any data we want to store (e.g. the model weights)
- The path to the `python_root` of the code along with relative paths for any dirs within the `python_root` we want to package
- An entrypoint function that returns a model given a path to the packaged data. See the docstring for `create_pytorch_neuropod` for more details and examples.
- The dependencies of our model. These should be python packages.

For our model:
- We don't need to store any data (because our model has no weights)
- Our python root is `/my/model/code/dir` and we want to store all the code in it
- Our entrypoint function is `get_model` and our entrypoint_package is `main` (because the code is in `main.py` in the python root)

This translates to the following. See the docstring for `create_pytorch_neuropod` for detailed descriptions of every parameter

```py
# `create_pytorch_neuropod` runs inference with the test data immediately
# after creating the neuropod. Raises a ValueError if the model output
# does not match the expected output.
create_pytorch_neuropod(
    neuropod_path=neuropod_path,
    model_name="addition_model",
    data_paths=[],
    code_path_spec=[{
        "python_root": '/my/model/code/dir',
        "dirs_to_package": [
            ""  # Package everything in the python_root
        ],
    }],
    entrypoint_package="main",
    entrypoint="get_model",
    input_spec=addition_problem_definition.INPUT_SPEC,
    output_spec=addition_problem_definition.OUTPUT_SPEC,
    test_input_data=addition_problem_definition.TEST_INPUT_DATA,
    test_expected_out=addition_problem_definition.TEST_EXPECTED_OUT,
    test_deps=['torch', 'numpy'],
)
```

### TorchScript

TorchScript is much easier to package than PyTorch (since we don't need to store any python code).

```py
class AdditionModel(torch.jit.ScriptModule):
    """
    A simple addition model
    """
    @torch.jit.script_method
    def forward(self, x, y):
        # dicts are not supported in TorchScript so we can't do this:
        # return {
        #     "out": x + y
        # }
        # Instead, we return the data as a tuple of key value pairs
        return (("out", x + y),)
```

To package this model, we can do the following:

```py
# `create_torchscript_neuropod` runs inference with the test data immediately
# after creating the neuropod. Raises a ValueError if the model output
# does not match the expected output.
create_torchscript_neuropod(
    neuropod_path=neuropod_path,
    model_name="addition_model",
    module=AdditionModel(),
    input_spec=addition_problem_definition.INPUT_SPEC,
    output_spec=addition_problem_definition.OUTPUT_SPEC,
    test_input_data=addition_problem_definition.TEST_INPUT_DATA,
    test_expected_out=addition_problem_definition.TEST_EXPECTED_OUT,
)
```

## Run Inference

Inference is the exact same no matter what the underlying DL framework is

### From Python

```py
x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])

with load_neuropod(ADDITION_MODEL_PATH) as neuropod:
  results = neuropod.infer({"x": x, "y": y})

  # array([6, 8, 10, 12])
  print results["out"]
```

### From C++

```cpp
const std::vector<int64_t> shape = {4};

// To show different usages of `add_tensor`, one of our inputs is a vector
// and the other is an array
const float[]            x_data = {1, 2, 3, 4};
const std::vector<float> y_data = {5, 6, 7, 8};

// Load the neuropod
Neuropod neuropod(ADDITION_MODEL_PATH);

// Get an input builder and add some data
auto input_builder = neuropod.get_input_builder();

// Add the input data using two different signatures of `add_tensor`
// (one with a pointer and size, one with a vector)
auto input_data    = input_builder->add_tensor("x", x_data, 4, shape)
                                   .add_tensor("y", y_data, shape)
                                   .build();

// Run inference
const auto output_data = neuropod.infer(input_data);

const auto out_tensor = output_data->find_or_throw("out");

// {6, 8, 10, 12}
const auto out_vector = out_tensor->as_typed_tensor<float>()->get_data_as_vector();

// {4}
const auto out_shape  = out_tensor->get_dims();
```

# Appendix

## Example Problem Definitions

The problem definition for 2d object detection may look something like this:

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
