load("//bazel:version.bzl", "NEUROPOD_VERSION")

# https://docs.bazel.build/versions/master/skylark/repository_rules.html
def _impl(repository_ctx):
    # The `or` pattern below handles empty strings and unset env variables
    # Using a default value only handles unset env variables
    version = repository_ctx.os.environ.get("NEUROPOD_TENSORFLOW_VERSION") or "1.12.0"
    IS_MAC = repository_ctx.os.name.startswith("mac")
    IS_GPU = (repository_ctx.os.environ.get("NEUROPOD_IS_GPU") or None) != None

    MAPPING = {
        # Linux CPU
        "1.12.0-linux-cpu": {
            "url": "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz",
            "sha256": "fd473e2ef72a446421f627aebe90479b92495965c26843034625677a14d8d64f",
        },
        "1.13.1-linux-cpu": {
            "url": "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.13.1.tar.gz",
            "sha256": "934c482948b64b31c037162e8485a6703814eee2393a0b9e54341a1d3ebed631",
        },
        "1.14.0-linux-cpu": {
            "url": "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.14.0.tar.gz",
            "sha256": "d30c15981d47c081ed4c38b5febf95b9260936fdc4a4b90b436b6982651c7111",
        },
        "1.15.0-linux-cpu": {
            "url": "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz",
            "sha256": "decbfd5a709eced3523f55ccfa239337a87e1ab3e000efda3617db79e1034ded",
        },
        # There isn't an official libtensorflow release for TF 2.x so I had to build this from source
        # TODO(vip): Update to an official release once there is one
        "2.2.0-linux-cpu": {
            "url": "https://github.com/VivekPanyam/tensorflow-prebuilts/releases/download/0.0.1/libtensorflow2.2.0rc3+-cpu-linux-x86_64.tar.gz",
            "sha256": "ee287e2ccd41e652f51e8c36370c841caf02115bf7b0cb7d4c2119b2c8cbd5fb",
        },
        "2.5.0-linux-cpu": {
            "url": "https://github.com/VivekPanyam/tensorflow-prebuilts/releases/download/0.0.1/libtensorflow2.5.0rc1-cpu-linux-x86_64.tar.gz",
            "sha256": "",
        },
        "2.6.2-linux-cpu": {
            # Note: they didn't release an updated libtensorflow for 2.6.2 (vs 2.6.0)
            "url": "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.6.0.tar.gz",
            "sha256": "58ff05f77aa1d969f912857e8a4edb17cc5b8220eeff13aaf807dcc10716a45d",
        },

        # Linux GPU
        "1.12.0-linux-gpu": {
            "url": "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.12.0.tar.gz",
            "sha256": "34e8070b391a2aa4317c786b80467e37b4d82fd8b880746a2775faeb7fd22d72",
        },
        "1.13.1-linux-gpu": {
            "url": "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.13.1.tar.gz",
            "sha256": "b7f1a8cdebaf8ff76cde7768ab350aa546893407336dc852dc425f07c903523c",
        },
        "1.14.0-linux-gpu": {
            "url": "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.14.0.tar.gz",
            "sha256": "a6972020d7cd1dfe53a3b2d4a867e4d6d8511bc15bfae8e4edfc4999a4bfd6b9",
        },
        "1.15.0-linux-gpu": {
            "url": "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.15.0.tar.gz",
            "sha256": "98e20336ff2b9acf2121b5c1ea00c35561ae732303bba8cfec167db3f7aea681",
        },
        # There isn't an official libtensorflow release for TF 2.x so I had to build this from source
        # TODO(vip): Update to an official release once there is one
        "2.2.0-linux-gpu": {
            "url": "https://github.com/VivekPanyam/tensorflow-prebuilts/releases/download/0.0.1/libtensorflow2.2.0rc3+-gpu-linux-x86_64.tar.gz",
            "sha256": "293f907623d7c9622d0ae9b17a2b5938c633eb8e32e9461346fe6afc36fe2e71",
        },
        "2.5.0-linux-gpu": {
            "url": "https://github.com/VivekPanyam/tensorflow-prebuilts/releases/download/0.0.1/libtensorflow2.5.0rc1-gpu-linux-x86_64.tar.gz",
            "sha256": "",
        },
        "2.6.2-linux-gpu": {
            # Note: they didn't release an updated libtensorflow for 2.6.2 (vs 2.6.0)
            "url": "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz",
            "sha256": "1a93057baa9f831a00a5935132c8e7438ee4ddfc166779dca51aae8c4a40870b",
        },

        # Mac CPU
        "1.12.0-mac-cpu": {
            "url": "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.12.0.tar.gz",
            "sha256": "0f77844966cfe8053eaa74f2b6bc5fecfddac070e4be49bd96ad70d5210dd8cc",
        },
        "1.13.1-mac-cpu": {
            "url": "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.13.1.tar.gz",
            "sha256": "b8b67823531b12cdbd0cf4bdaf97d207a5821541b8dd67a4a1e1a0356dee6057",
        },
        "1.14.0-mac-cpu": {
            "url": "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.14.0.tar.gz",
            "sha256": "22c10ca842ef65018fae762dffe733843b537ec08701f8a59127a8a0692c4d7f",
        },
        "1.15.0-mac-cpu": {
            "url": "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.15.0.tar.gz",
            "sha256": "1a9da42e31f613f1582cf996e3ead32528964994eb98b7c355923f2dc39bfce0",
        },
        # There isn't an official libtensorflow release for TF 2.x so I had to build this from source
        # TODO(vip): Update to an official release once there is one
        "2.2.0-mac-cpu": {
            "url": "https://github.com/VivekPanyam/tensorflow-prebuilts/releases/download/0.0.1/libtensorflow2.2.0rc3+-cpu-darwin-x86_64.tar.gz",
            "sha256": "78e788a9ada69c8b6edd4061b21ee2a87dbcc41e0c55893c5b063140c28f969d",
        },
        "2.5.0-mac-cpu": {
            "url": "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-2.5.0.tar.gz",
            "sha256": "",
        },
        "2.6.2-mac-cpu": {
            # Note: they didn't release an updated libtensorflow for 2.6.2 (vs 2.6.0)
            "url": "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-2.6.0.tar.gz",
            "sha256": "61c5478fa9ca813e5f30c536240bdbb77f1e5fd3d4687211baab84ecfe4bb3b9",
        },
    }

    download_mapping = MAPPING["{}-{}-{}".format(
        version,
        "mac" if IS_MAC else "linux",
        "gpu" if IS_GPU else "cpu",
    )]

    download_url = download_mapping["url"]
    sha256 = download_mapping["sha256"]

    repository_ctx.download_and_extract(download_url, sha256 = sha256)
    repository_ctx.symlink(repository_ctx.path(Label(repository_ctx.attr.build_file)), "BUILD.bazel")

    # Create a file that specifies versioning information
    repository_ctx.file(
        "neuropod_backend_path.bzl",
        content = """
        NEUROPOD_BACKEND_PATH = "{}/backends/tensorflow_{}{}/"
        """.format(
            NEUROPOD_VERSION,
            version,
            "_gpu" if IS_GPU else "",
        ).strip(),
    )

tensorflow_repository = repository_rule(
    implementation = _impl,
    environ = [
        "NEUROPOD_TENSORFLOW_VERSION",
        "NEUROPOD_IS_GPU",
    ],
    attrs = {"build_file": attr.string(mandatory = True)},
)
