# Python Guide

This guide walks through loading a Neuropod and running inference from Python

!!! tip
    The Neuropod runtime interface is identical for all frameworks so this guide applies for models from all supported frameworks (including TensorFlow, PyTorch, Keras, and TorchScript)


## Packaging a Neuropod

See the [basic introduction guide](tutorial.md) for examples of how to create Neuropod models in all the supported frameworks.

## Loading a Neuropod

```py
from neuropod.loader import load_neuropod

neuropod = load_neuropod(PATH_TO_MY_MODEL)
```

You can also use `load_neuropod` as a context manager:

```py
from neuropod.loader import load_neuropod

with load_neuropod(PATH_TO_MY_MODEL) as neuropod:
    # Do something here
    pass
```

### Options

You can also provide runtime options when loading a model.

To select what device to run the model on, you can supply a `visible_gpu` argument.

This is the index of the GPU that this Neuropod should run on (if any). It can either be `None` or a nonnegative integer.
Setting this to `None` will attempt to run this model on CPU.

```py
# Run on CPU
neuropod = load_neuropod(PATH_TO_MY_MODEL, visible_gpu=None)

# Run on the second GPU
neuropod = load_neuropod(PATH_TO_MY_MODEL, visible_gpu=1)
```

### Get the inputs and outputs of a model

The inputs and outputs of a model are available via the `inputs` and `outputs` property.

```py
with load_neuropod(PATH_TO_MY_MODEL) as neuropod:
    # This is a list of dicts containing the "name", "dtype", and "shape"
    # of the input
    print(neuropod.inputs, neuropod.outputs)
```

## Inference
The `infer` method of a model is used to run inference. The input to this method is a dict mapping input names to values. This must match the input spec in the neuropod config for the loaded model.

!!! note
    All the keys in this dict must be strings and all the values must be numpy arrays

The output of `infer` is a dict mapping output names to values. This is checked to ensure that it matches the spec in the neuropod config for the loaded model. All the keys in this dict are strings and all the values are numpy arrays.


```py
x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])

with load_neuropod(ADDITION_MODEL_PATH) as neuropod:
  results = neuropod.infer({"x": x, "y": y})

  # array([6, 8, 10, 12])
  print results["out"]
```

## Serialization

```py
import numpy as np
from neuropod import neuropod_native

# An array to serialize
tensor = np.arange(5)

# Convert a numpy array to a NeuropodTensor and serialize it
serialized_bytes = neuropod_native.serialize(tensor)

# Deserialize a string of bytes to a NeuropodTensor
# (and return it as a numpy array)
deserialized = neuropod_native.deserialize(serialized_bytes)

# array([0, 1, 2, 3, 4])
print(deserialized)
```

Under the hood, the serialization code converts between numpy arrays and C++ NeuropodTensor objects (in a zero-copy way). It then uses the C++ serialization functionality to serialize/deserialize.

!!! note
    Serialization and deserialization works across Python and C++. This means you can serialize tensors in C++ and deserialize in Python or vice-versa

!!! warning
    The goal for this API is to support transient serialization. There are no guarantees about backwards compatibility so this API should not be used for long term storage of data
