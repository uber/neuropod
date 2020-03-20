# Out-of-Process Execution

Neuropod can run models in different processes using an optimized shared memory implementation with extremely low overhead (~100 to 500 microseconds).


To run a model in another process, set the `use_ope` option when loading a model:

```cpp
neuropod::RuntimeOptions opts;
opts.use_ope = true;
Neuropod model(neuropod_path, opts);
```

Nothing else should need to change.

There are many potential benefits of this approach:

- Run models that require different versions of Torch or TF from the same "master" process
- Pin the worker process to a specific core to reduce variability in inference time
- Isolate models from each other and from the rest of your program


The worker process can also be run in a docker container to provide even more isolation.


For more details and options, see the `OPEOptions` struct inside `RuntimeOptions`.

TODO(vip): OPE string tensor support
