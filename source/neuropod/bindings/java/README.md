# Java API

***NOTE: This API is currently experimental***
#  Build jar
```shell script
bazel build --verbose_failures --action_env=NEUROPOD_TENSORFLOW_VERSION=1.15.0 --action_env=NEUROPOD_TORCH_VERSION=1.4.0 //neuropod/bindings/java:neuropod_java
```

#  Build fat jar
Fat jar is self-contained jar that includes native OS-Arch specific files: JNI library, Neuropod Core and Backend packages.
```shell script
bazel build --verbose_failures --action_env=NEUROPOD_TENSORFLOW_VERSION=1.15.0 --action_env=NEUROPOD_TORCH_VERSION=1.4.0 //neuropod/bindings/java:neuropod_java_jar
```

# Run Java tests only
```shell script
export NEUROPOD_BASE_DIR=/tmp/.neuropod_test_base
rm -rf "/tmp/.neuropod_test_base" && mkdir "/tmp/.neuropod_test_base"
bazel test --action_env=NEUROPOD_TENSORFLOW_VERSION=1.15.0 --action_env=NEUROPOD_TORCH_VERSION=1.4.0 --test_lang_filters="java" --test_tag_filters="-gpu" //...
```

Note that test builds fat jar first, that has "neuropod standard" directory structure
```text
0.2.0/backends/tensorflow_1.15.0/
0.2.0/backends/torchscript_1.4.0/
```
that is expected by "register backends" logic.

## Usage

```java
// Load a model.
Neuropod model = new Neuropod(modelPath);

// Get an allocator from the model.
NeuropodTensorAllocator allocator = model.getTensorAllocator();

// Build input data.
Map<String, NeuropodTensor> inputs = new HashMap<>();

// This will be a reference to output after successful inference.
Map<String, NeuropodTensor> valueMap = null;

// We may declare one or more resources in a try-with-resources statement.
try (
    NeuropodTensor tensorX = allocator.create(new float[]{1.0f, 3.0f}, Arrays.asList(2L, 1L), model);
    NeuropodTensor tensorY = allocator.create(new float[]{2.0f, 5.0f}, Arrays.asList(2L, 1L), model);
) {
    inputs.put("request_location_latitude", tensorX);
    inputs.put("request_location_longitude", tensorY);

    // Run inference. Infer may throw exception if failed.
    valueMap = neuropod.infer(inputs);
}

// Get the output.
NeuropodTensor output = valueMap.get(key);
FloatBuffer res = output.toFloatBuffer();

// Do something with the output.

// Clean up other Neuropod resources.
allocator.close();
output.close();
model.close();
```

Note: Using [try-with-resources](https://docs.oracle.com/javase/tutorial/essential/exceptions/tryResourceClose.html) can help improve exception safety of your code.
This is useful in case of multiple Tensor allocations, as explained at https://docs.oracle.com/javase/specs/jls/se8/html/jls-14.html#jls-14.20.3
"Resources are initialized in left-to-right order. If a resource fails to initialize (that is, its initializer expression throws an exception), then all resources initialized so far by the try-with-resources statement are closed."
