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

See the Backends section below for instructions on installing backends.

## C++

Prebuilts can be downloaded from the [releases](https://github.com/uber/neuropod/releases) page.

The `libneuropod-[os]-[version].tar.gz` files contain header files and prebuilt binaries for the main Neuropod library.

To run models, you must also install packages for "backends". These are fully self-contained packages that let Neuropod run models
with specific versions of frameworks regardless of the version installed on your system.

See the Backends section below for instructions on installing backends.


## Backends

The following commands can be used to install the official backends. Backends implement support for a particular framework within Neuropod (e.g. Torch 1.7.0 on GPU, TensorFlow 1.15.0 on CPU, etc) Once a backend is installed, Neuropod can use it from any supported language.

```sh
# Create a folder to store backends.
# The location of the folder Neuropod expects backends to be installed into defaults to "/usr/local/lib/neuropod",
# but can be overridden by setting the NEUROPOD_BASE_DIR environment variable at runtime
NEUROPOD_BASE_DIR="/usr/local/lib/neuropod"
sudo mkdir -p "$NEUROPOD_BASE_DIR"

# Find URLs of backends you want to install from the releases page (https://github.com/uber/neuropod/releases) and install them
# by untarring them in your NEUROPOD_BASE_DIR directory.
# For example, to install a GPU enabled Torch 1.7 backend for CUDA 10.1, run
curl -L https://github.com/uber/neuropod/releases/download/v0.3.0-rc7/libneuropod-gpu-cuda-10.1-linux-v0.3.0-rc7-torchscript-1.7.0-backend.tar.gz | sudo tar -xz -C "$NEUROPOD_BASE_DIR"
```

Multiple backends can be installed for a given framework and Neuropod will select the correct one when loading a model.
An error will be thrown if none of the installed backends match the model's requirements.

See the [basic introduction](tutorial.md) for more information on getting started.
