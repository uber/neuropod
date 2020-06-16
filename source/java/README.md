# To build this neuropod java api test under OS X environment:
1. Build the cpp neuropod
2. Make sure the g++ can find the header and library of neuropod, for example copy them to /usr/local/include/ and /usr/local/lib/ respectively, and change the NEUROPOD_PATH and JAVA_HOME to the correct path.
3. To run the simple test:
```
make run
```
4. To clean up:
```
make clean
```

# The java api interface:
1. To load a model:
```
Neuropod neuropod = new Neuropod(modelPath);
```
2. To prepare the input feature:
```
Map<String, Object> inputs = new HashMap<>();
// Object should be a list of numeric objects or strings. If the input is a tensor, it should be flattened to a List first.
```
3. To do the inference job:
```
NeuropodValueMap valueMap = neuropod.infer(inputs);
```

4. To retrieve the results:
```
List<Object> results = valueMap.getValue(key).toList();
```

5. Don't forget to clean up the memory:
```
valueMap.close();
neuropod.close();
```
