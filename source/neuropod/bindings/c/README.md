# C API

***NOTE: This API is currently experimental***

## Build
```
bazel build //neuropod/bindings/c:c_api
```

## Run C unit tests
```
bazel test --test_output=errors //neuropod/tests:test_c_api
```
