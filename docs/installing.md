# Installing Neuropod

!!! note
    Neuropod requires macOS or Linux.

## Python

Neuropod can be installed using pip:

```sh
pip install neuropod
```

To run models, you must also install packages for "backends". These are fully self-contained packages that let Neuropod run models
with specific versions of frameworks regardless of the version installed in your python environment.

The following commands can be used to install the official backends:

```sh
# Torch CPU
pip install neuropod-backend-torchscript-1-1-0-cpu
pip install neuropod-backend-torchscript-1-2-0-cpu
pip install neuropod-backend-torchscript-1-3-0-cpu
pip install neuropod-backend-torchscript-1-4-0-cpu

# Torch GPU
pip install neuropod-backend-torchscript-1-1-0-gpu-cuda-9-0
pip install neuropod-backend-torchscript-1-2-0-gpu-cuda-10-0
pip install neuropod-backend-torchscript-1-3-0-gpu-cuda-10-0
pip install neuropod-backend-torchscript-1-4-0-gpu-cuda-10-0
pip install neuropod-backend-torchscript-1-5-0-gpu-cuda-10-1

# TF CPU
pip install neuropod-backend-tensorflow-1-12-0-cpu
pip install neuropod-backend-tensorflow-1-13-1-cpu
pip install neuropod-backend-tensorflow-1-14-0-cpu
pip install neuropod-backend-tensorflow-1-15-0-cpu

# TF GPU
pip install neuropod-backend-tensorflow-1-12-0-gpu-cuda-9-0
pip install neuropod-backend-tensorflow-1-13-1-gpu-cuda-10-0
pip install neuropod-backend-tensorflow-1-14-0-gpu-cuda-10-0
pip install neuropod-backend-tensorflow-1-15-0-gpu-cuda-10-0

# Python
pip install neuropod-backend-python-27
pip install neuropod-backend-python-35
pip install neuropod-backend-python-36
pip install neuropod-backend-python-37
pip install neuropod-backend-python-38
```

Multiple backends can be installed for a given framework and Neuropod will select the correct one when loading a model.
An error will be thrown if none of the installed backends match the model's requirements.

## C++

Prebuilts can be downloaded from the [releases](https://github.com/uber/neuropod/releases) page.

The `libneuropod-[os]-[version].tar.gz` files contain the main Neuropod library.

The rest of the tar files contain the backends. These can be installed by adding them to your library path or directly linking your application to them.

!!! note
    For the C++ interface, currently, only one version for each framework can be installed at a time. This is temporary with a more stable installation process coming soon.


See the [basic introduction](tutorial.md) for more information on getting started.
