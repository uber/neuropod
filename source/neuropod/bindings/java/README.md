# The java api interface:

1. To build & to unit test:
```
bazel build //neuropod/bindings/java:neuropod_java_jar
bazel test //neuropod/bindings/java:*
```

2. To load a model:
```
Neuropod model = new Neuropod(modelPath);
```
3. To prepare the input feature:
```
Map<String, NeuropodTensor> inputs = new HashMap<>();
NeuropodTensorAllocator allocator = model.getTensorAllocator();
NeuropodTensor tensor1 = allocator.create(new float[]{1.0f, 3.0f}, Arrays.asList(2L, 1L), model);
inputs.put("request_location_latitude",tensor1);
tensor1.close();
NeuropodTensor tensor2 = allocator.create(new float[]{2.0f, 5.0f}, Arrays.asList(2L, 1L), model);
inputs.put("request_location_longitude",tensor2);
tensor2.close();
...
```
4. To do the inference job:
```
Map<String, NeuropodTensor> valueMap = neuropod.infer(inputs);
```

5. To retrieve the results:
```
NeuropodTensor output = valueMap.get(key);
FloatBuffer res = output.toFloatBuffer();
```

6. Don't forget to clean up the memory:
```
allocator.close();
output.close();
valueMap.close();
model.close();
```
