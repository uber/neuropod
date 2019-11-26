# https://docs.bazel.build/versions/master/skylark/repository_rules.html
def _impl(repository_ctx):
    # The `or` pattern below handles empty strings and unset env variables
    # Using a default value only handles unset env variables
    version = repository_ctx.os.environ.get("NEUROPODS_TORCH_VERSION") or "1.1.0"
    IS_MAC  = repository_ctx.os.name.startswith("mac")
    IS_GPU  = (repository_ctx.os.environ.get("NEUROPODS_IS_GPU") or None) != None
    CUDA_VERSION = repository_ctx.os.environ.get("NEUROPODS_CUDA_VERSION") or "10.0"

    # Get the torch cuda string (e.g. cpu, cu90, cu92, cu100)
    torch_cuda_string = "cu" + CUDA_VERSION.replace(".", "") if IS_GPU else "cpu"

    defines = []

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
        "1.3.0.dev20190820-linux-cpu": {
            "url": "https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-1.3.0.dev20190820%2Bcpu.zip",
            "sha256": "8f6a24af454bdcd5782251676e5af8c6e8ce666fbd01c66eeb0d251900e71f5c",
        },
        "1.3.0-linux-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.3.0%2Bcpu.zip",
            "sha256": "a1a4bfe2090c418150cf38b37e43b3238b9639806f0c3483097d073792c2e114",
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
        "1.3.0.dev20190820-linux-cu100": {
            "url": "https://download.pytorch.org/libtorch/nightly/cu100/libtorch-shared-with-deps-1.3.0.dev20190820.zip",
            "sha256": "538cde5b08826a38f7686809a18419f75d79918c4c48bb6f7c107776f27398a1",
        },
        "1.3.0-linux-cu100": {
            "url": "https://download.pytorch.org/libtorch/cu100/libtorch-shared-with-deps-1.3.0.zip",
            "sha256": "5943ed9d25f473f9baf4301fc6526f048f89061f38e8cf0cc01506c96ad58ed4",
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
        "1.3.0.dev20190820-mac-cpu": {
            "url": "https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-1.3.0.dev20190820.zip",
            "sha256": "e809c1948a11c7361ff9aa500100cca8db7eaebc76dae8bd22304390b2effc9e",
        },
        "1.3.0-mac-cpu": {
            "url": "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.3.0.zip",
            "sha256": "c44050d28bf21676f68fa0f87caa27bc610cd9802c41b5c83e87295d22e048a4",
        },
    }

    download_mapping = MAPPING["{}-{}-{}".format(
        version,
        "mac" if IS_MAC else "linux",
        torch_cuda_string
    )]

    download_url = download_mapping["url"]
    sha256 = download_mapping["sha256"]

    repository_ctx.download_and_extract(download_url, stripPrefix="libtorch", sha256=sha256)

    # Generate a build file based on the template
    repository_ctx.template(
        "BUILD.bazel",
        repository_ctx.path(Label(repository_ctx.attr.build_file_template)),
        substitutions = {
            "{TORCH_DEFINES}": "{}".format(defines),
        },
    )

libtorch_repository = repository_rule(
    implementation=_impl,
    local=True,
    attrs={"build_file_template": attr.string(mandatory=True)})
