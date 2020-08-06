# The C api interface:

1. To build &to unit test :
``` 
bazel build // neuropod/bindings/c:c_api
bazel test --test_output=errors // neuropod/tests:test_c_api
```
