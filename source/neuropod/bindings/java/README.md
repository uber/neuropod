# Java API

***NOTE: This API is currently experimental***

## Usage

```java
// Load a model
Neuropod model = new Neuropod(modelPath);

// Get an allocator from the model
NeuropodTensorAllocator allocator = model.getTensorAllocator();

// Build input data
Map<String, NeuropodTensor> inputs = new HashMap<>();

try (NeuropodTensor tensor = allocator.create(new float[]{1.0f, 3.0f}, Arrays.asList(2L, 1L), model)) {
    inputs.put("request_location_latitude", tensor);
}

try (NeuropodTensor tensor = allocator.create(new float[]{2.0f, 5.0f}, Arrays.asList(2L, 1L), model)) {
    inputs.put("request_location_longitude", tensor);
}

// Run inference
Map<String, NeuropodTensor> valueMap = neuropod.infer(inputs);

// Get the output
NeuropodTensor output = valueMap.get(key);
FloatBuffer res = output.toFloatBuffer();

// Do something with the output

// Clean up
allocator.close();
output.close();
valueMap.close();
model.close();
```

Note: Using [try-with-resources](https://docs.oracle.com/javase/tutorial/essential/exceptions/tryResourceClose.html) can help improve exception safety of your code.
