load("//bazel:version.bzl", "NEUROPOD_VERSION")

# https://docs.bazel.build/versions/master/skylark/repository_rules.html
def _impl(repository_ctx):
    # The `or` pattern below handles empty strings and unset env variables
    # Using a default value only handles unset env variables
    version = repository_ctx.os.environ.get("NEUROPOD_TORCH_VERSION") or "1.1.0"
    IS_MAC = repository_ctx.os.name.startswith("mac")
    IS_GPU = (repository_ctx.os.environ.get("NEUROPOD_IS_GPU") or None) != None
    CUDA_VERSION = repository_ctx.os.environ.get("NEUROPOD_CUDA_VERSION") or "10.0"

    # TODO(vip): Fix this once we have a better way of dealing with CUDA 11.2
    if CUDA_VERSION == "11.2.1":
        CUDA_VERSION = "10.1"
        IS_GPU = False

    # Get the torch cuda string (e.g. cpu, cu90, cu92, cu100)
    torch_cuda_string = "cu" + CUDA_VERSION.replace(".", "") if IS_GPU else "cpu"

    defines = ["TORCH_VERSION=" + version]

    # If this is a nightly build, we want to define a variable
    # to let our code know what nightly version this is
    # See https://github.com/pytorch/pytorch/issues/23094
    if "dev" in version:
        version_base, version_date = version.split(".dev")
        defines.append("CAFFE2_NIGHTLY_VERSION=" + version_date)

    MAPPING = {
        # Linux CPU
        "1.1.0-linux-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.1.0.zip",
            "sha256": "c863a0073ff4c7b6feb958799c7dc3202b3449e86ff1cec9c85c7da9d1fe0218",
        },
        "1.2.0-linux-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.2.0.zip",
            "sha256": "6b0cc8840e05e5e2742e5c59d75f8379f4eda8737aeb24b5ec653735315102b2",
        },
        "1.3.0-linux-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.3.0%2Bcpu.zip",
            "sha256": "a1a4bfe2090c418150cf38b37e43b3238b9639806f0c3483097d073792c2e114",
        },
        "1.4.0-linux-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.4.0%2Bcpu.zip",
            "sha256": "cf2d79574e08198419fd53d3b0edab3e12587649a22185431e3f5c8937177a47",
        },
        "1.5.0-linux-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.5.0%2Bcpu.zip",
            "sha256": "db3545b0d2b144db4292c2f0bec236febec44aa658dd54f6b3532f2848c50c8a",
        },
        "1.6.0-linux-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.6.0%2Bcpu.zip",
            "sha256": "31d3c5a59b1394f9d958501e392cedc91476358accea76c7094d103e9335b80c",
        },
        "1.7.0-linux-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.7.0%2Bcpu.zip",
            "sha256": "4cf8635fb41774c3b38fcb9955ff86b4ed7eb8e73d1595a09297196b7c28db28",
        },
        "1.8.1-linux-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.8.1%2Bcpu.zip",
            "sha256": "b2df0393b3a5445e4e644729c6e0610437af983ddea4b0f5c46e01651a64bd74",
        },
        "1.9.0-linux-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.9.0%2Bcpu.zip",
            "sha256": "",
        },
        "1.10.2-linux-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.10.2%2Bcpu.zip",
            "sha256": "fa3fad287c677526277f64d12836266527d403f21f41cc2e7fb9d904969d4c4a",
        },

        # Linux GPU
        "1.1.0-linux-cu90": {
            "url": "https://download.pytorch.org/libtorch/cu90/libtorch-shared-with-deps-1.1.0.zip",
            "sha256": "57ff5faa79c9729f35a2a753717abcef8096cc5646a7b79ddcef2288be5281a9",
        },
        "1.2.0-linux-cu100": {
            "url": "https://download.pytorch.org/libtorch/cu100/libtorch-shared-with-deps-1.2.0.zip",
            "sha256": "bd385169dd6137f532648398eeee8d6479be1f6b81314a4373800fcc72bb375d",
        },
        "1.3.0-linux-cu100": {
            "url": "https://download.pytorch.org/libtorch/cu100/libtorch-shared-with-deps-1.3.0.zip",
            "sha256": "5943ed9d25f473f9baf4301fc6526f048f89061f38e8cf0cc01506c96ad58ed4",
        },
        "1.4.0-linux-cu100": {
            "url": "https://download.pytorch.org/libtorch/cu100/libtorch-shared-with-deps-1.4.0%2Bcu100.zip",
            "sha256": "1557927c9929c8eb8caf8860d0ffdce39ae931af924f0fde859ad1dc0843575c",
        },
        "1.5.0-linux-cu101": {
            "url": "https://download.pytorch.org/libtorch/cu101/libtorch-shared-with-deps-1.5.0%2Bcu101.zip",
            "sha256": "04c0fdb46ca1b74c39715d735a4906d08b976f1d57aef31a020eaf967a6a48b7",
        },
        "1.6.0-linux-cu101": {
            "url": "https://download.pytorch.org/libtorch/cu101/libtorch-shared-with-deps-1.6.0%2Bcu101.zip",
            "sha256": "6a5a215da3dff3f3183674187de3290bd66be3f0ca686ea61f3dd34530da8e23",
        },
        "1.7.0-linux-cu101": {
            "url": "https://download.pytorch.org/libtorch/cu101/libtorch-shared-with-deps-1.7.0%2Bcu101.zip",
            "sha256": "e0816a692e4540739b6832c118f186b5e65d1fac56edb2048b600b756ae42687",
        },
        "1.8.1-linux-cu102": {
            "url": "https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.8.1%2Bcu102.zip",
            "sha256": "b1d82045bba9f69752165ed46dbe1996ecc646a4ff31ce5883192d4b52a84f3e",
        },
        "1.9.0-linux-cu102": {
            "url": "https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.9.0%2Bcu102.zip",
            "sha256": "",
        },
        "1.10.2-linux-cu102": {
            "url": "https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.10.2%2Bcu102.zip",
            "sha256": "206ab3f44d482a1d9837713cafbde9dd9d7907efac2dc94f1dc86e9a1101296f",
        },

        # Mac CPU
        "1.1.0-mac-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.1.0.zip",
            "sha256": "2db31f6c7e69ea9142396d8ed0a7ad70dde2a9993cc8c23cc48c03ffeea13f0f",
        },
        "1.2.0-mac-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.2.0.zip",
            "sha256": "927cd63106d4055d4a415cf75b2ecffb430c27736b78f609350b57934883240f",
        },
        "1.3.0-mac-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.3.0.zip",
            "sha256": "c44050d28bf21676f68fa0f87caa27bc610cd9802c41b5c83e87295d22e048a4",
        },
        "1.4.0-mac-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.4.0.zip",
            "sha256": "84e9112b442ee1e3dc9e078d9066a855a2344ec566616cffbff1662e08cd8bf7",
        },
        "1.5.0-mac-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.5.0.zip",
            "sha256": "90bd7e5df2a73af1d80cdaa1403b6f5cc5ac9127be4bb5b7616bf32a868cf7d8",
        },
        "1.6.0-mac-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.6.0.zip",
            "sha256": "e1140bed7bd56c26638bae28aebbdf68e588a9fae92c6684645bcdd996e4183c",
        },
        "1.7.0-mac-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.7.0.zip",
            "sha256": "2dccb83f1beb16ef3129d9f7af5abafb2f5d9220d0c7da8fde3531bd4ef0e655",
        },
        "1.8.1-mac-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.8.1.zip",
            "sha256": "",
        },
        "1.9.0-mac-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.9.0.zip",
            "sha256": "",
        },
        "1.10.2-mac-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.10.2.zip",
            "sha256": "d1711e844dc69c2338adfc8ce634806a9ae36e54328afbe501bafd2d70f550e2",
        },
    }

    download_mapping = MAPPING["{}-{}-{}".format(
        version,
        "mac" if IS_MAC else "linux",
        torch_cuda_string,
    )]

    download_url = download_mapping["url"]
    sha256 = download_mapping["sha256"]

    repository_ctx.download_and_extract(download_url, stripPrefix = "libtorch", sha256 = sha256)

    # Generate a build file based on the template
    repository_ctx.template(
        "BUILD.bazel",
        repository_ctx.path(Label(repository_ctx.attr.build_file_template)),
        substitutions = {
            "{TORCH_DEFINES}": "{}".format(defines),
        },
    )

    # Create a file that specifies versioning information
    repository_ctx.file(
        "neuropod_backend_path.bzl",
        content = """
        NEUROPOD_BACKEND_PATH = "{}/backends/torchscript_{}{}/"
        """.format(
            NEUROPOD_VERSION,
            version,
            "_gpu" if IS_GPU else "",
        ).strip(),
    )

libtorch_repository = repository_rule(
    implementation = _impl,
    environ = [
        "NEUROPOD_TORCH_VERSION",
        "NEUROPOD_IS_GPU",
        "NEUROPOD_CUDA_VERSION",
    ],
    attrs = {"build_file_template": attr.string(mandatory = True)},
)
