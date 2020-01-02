# Out-of-Process Execution

Neuropods can run models in different processes using an optimized shared memory implementation with extremely low overhead (~100 to 500 microseconds).


To run a model in another process, modify your loading code as follows:

```cpp
#include <neuropods/multiprocess/multiprocess.hh>

...

auto neuropod = neuropod::load_neuropod_in_new_process(neuropod_path);
```

Nothing else should need to change.

There are many potential benefits of this approach:

- Run models that require different versions of Torch or TF from the same "master" process
- Pin the worker process to a specific core to reduce variability in inference time
- Isolate models from each other and from the rest of your program


The worker process can also be run in a docker container to provide even more isolation.


TODO(vip): OPE string tensor support
