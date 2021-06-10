The code in this folder is necessary because we can't use the built-in `LoadSavedModel` in TensorFlow.

TF doesn't provide a good way to run graphs on different devices in the same process
For example, we can't use GPUOptions::visible_device_list as it is a per process setting

From: https://github.com/tensorflow/tensorflow/issues/18861#issuecomment-385610497
> Unfortunately, though `visible_deivces_list` is included in `ConfigProto`, it is
> actually a per-process setting. In fact, this is true of almost all options inside
> the GPUOptions protocol buffer.

The issue linked above goes through several other alternatives that don't work either.

Instead, we have to set the device of the graph by moving each node in the `GraphDef` to the target device before creating a session.

`LoadSavedModel` creates a session and loads a `GraphDef` internally so we can't use it directly.

The code in this folder is a port of saved model loader code in TF r2.4 with the `GraphDef` modification
described above.

All the code in this folder is based on the code from https://github.com/tensorflow/tensorflow/tree/r2.4/tensorflow/cc/saved_model

There are also a few minor changes to support multiple TF versions, remove some unneeded code, and remove some tensorflow-internal profiling code.