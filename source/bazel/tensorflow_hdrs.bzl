# https://docs.bazel.build/versions/master/skylark/repository_rules.html
def _impl(repository_ctx):
    # The `or` pattern below handles empty strings and unset env variables
    # Using a default value only handles unset env variables
    version = repository_ctx.os.environ.get("NEUROPOD_TENSORFLOW_VERSION") or "1.12.0"
    IS_MAC  = repository_ctx.os.name.startswith("mac")
    IS_GPU  = (repository_ctx.os.environ.get("NEUROPOD_IS_GPU") or None) != None

    MAPPING = {
        # Linux CPU
        "1.12.0-linux-cpu": {
            "url": "https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.12.0-cp27-none-linux_x86_64.whl",
            "sha256": "6064265fd16af798b46a83f42ea2ebc8c9d6710afa143a0f244ac42c40ab9fcd",
        },
        "1.13.1-linux-cpu": {
            "url": "https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.13.1-cp27-none-linux_x86_64.whl",
            "sha256": "8f5da949e3a56f6440d58281ebad20468325104173e1325a91927557fc683b78",
        },
        "1.14.0-linux-cpu": {
            "url": "https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.14.0-cp27-none-linux_x86_64.whl",
            "sha256": "f374ca80599d785c23ed467ec668ea0e9caf327857af08fb0ab09d345d6edc7b",
        },
        "1.15.0-linux-cpu": {
            "url": "https://files.pythonhosted.org/packages/ec/98/f968caf5f65759e78873b900cbf0ae20b1699fb11268ecc0f892186419a7/tensorflow-1.15.0-cp27-cp27mu-manylinux2010_x86_64.whl",
            "sha256": "af57e0e16adb4d6ccd387954c1d70e34cc4925b74da9135d2b83ca7d3dd9d102",
        },

        # Linux GPU
        "1.12.0-linux-gpu": {
            "url": "https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.12.0-cp27-none-linux_x86_64.whl",
            "sha256": "435a9a4a37c1a92f9bc80f577f0328775539c593b9bc9e943712a204ada11db5",
        },
        "1.13.1-linux-gpu": {
            "url": "https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.13.1-cp27-none-linux_x86_64.whl",
            "sha256": "ba010b08c19903cbc183012379e60755af7713c5ad138887c64bdb255c1ac22d",
        },
        "1.14.0-linux-gpu": {
            "url": "https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.14.0-cp27-none-linux_x86_64.whl",
            "sha256": "8563b56388f6e9efdca10fcea2c9f6f03b3e4c65cd22fa5a93d73caf5e483d78",
        },
        "1.15.0-linux-gpu": {
            "url": "https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.15.0-cp27-cp27mu-manylinux2010_x86_64.whl",
            "sha256": "829c90021ec0fa33d74c2bcbff4e1e365fc63e875d2f04b60451c5abfeac8382",
        },

        # Mac CPU
        "1.12.0-mac-cpu": {
            "url": "https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py2-none-any.whl",
            "sha256": "5cee35f8a6a12e83560f30246811643efdc551c364bc981d27f21fbd0926403d",
        },
        "1.13.1-mac-cpu": {
            "url": "https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.13.1-py2-none-any.whl",
            "sha256": "0f305f3c461ed2ce5e0b65fccc7b7452f483c7935dd8a52a466d622e642fdea8",
        },
        "1.14.0-mac-cpu": {
            "url": "https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.14.0-py2-none-any.whl",
            "sha256": "46fc216db780b1a7b7a28cf6536d6ec0a171ef2fa546a24934ffe077c734c891",
        },
        "1.15.0-mac-cpu": {
            "url": "https://files.pythonhosted.org/packages/b9/be/140e6c4deef96ddfe3837ef7ffc396a06cca73c958989835ac8f05773678/tensorflow-1.15.0-cp27-cp27m-macosx_10_11_x86_64.whl",
            "sha256": "0a01def34c28298970dc83776dd43877fd59e43fddd8e960d01b6eb849ba9938",
        },
    }

    download_mapping = MAPPING["{}-{}-{}".format(
        version,
        "mac" if IS_MAC else "linux",
        "gpu" if IS_GPU else "cpu"
    )]

    download_url = download_mapping["url"]
    sha256 = download_mapping["sha256"]

    repository_ctx.download_and_extract(download_url, type="zip", sha256=sha256)

    repository_ctx.template(
        "BUILD.bazel",
        repository_ctx.path(Label(repository_ctx.attr.build_file_template)),
        substitutions = {
            "{TENSORFLOW_VERSION}": version,
        },
    )

tensorflow_hdrs_repository = repository_rule(
    implementation=_impl,
    local=True,
    attrs={"build_file_template": attr.string(mandatory=True)})
