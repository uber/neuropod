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

    # The base version of torch (e.g. 1.2.0)
    version_base = None

    # If this is a nightly build, what's the date (e.g. 20190809)
    version_date = None

    # The base download url
    download_url = "https://download.pytorch.org/libtorch"
    defines = []

    # Get the version info
    if "dev" in version:
        version_base, version_date = version.split(".dev")
    else:
        version_base = version

    # If this is a nightly build, we want to define a variable
    # to let our code know what nightly version this is
    # See https://github.com/pytorch/pytorch/issues/23094
    if version_date != None:
        download_url += "/nightly"
        defines.append("CAFFE2_NIGHTLY_VERSION=" + version_date)

    # Mac builds do not have the cuda string as part of the version
    if not IS_MAC:
        # If this is a nightly build after they started adding the cuda string to the packages
        if version_date != None and int(version_date) > 20190723:
            if version_base == "1.3.0" and torch_cuda_string == "cu100":
                # At some point between 20190723 and 20190820, they stopped adding
                # cu100 to the build versions...
                pass
            else:
                version += "%2B" + torch_cuda_string

        # If this is the 1.2.0 stable release
        if version_base == "1.2.0" and version_date == None:
            # The initial version of the 1.2.0 stable release was broken
            # The manual rebuilds of this release don't include the CUDA version in the URLs
            # See https://github.com/pytorch/pytorch/issues/24120
            pass

        # If this is the 1.3.0 stable release
        if version_base == "1.3.0" and version_date == None and torch_cuda_string != "cu100":
            version += "%2B" + torch_cuda_string

    # Add the CUDA variant to the URL
    download_url += "/" + torch_cuda_string

    if IS_MAC:
        download_url += "/libtorch-macos-" + version + ".zip"
    else:
        download_url += "/libtorch-shared-with-deps-" + version + ".zip"

    # To prevent redownloading Torch during local development, we'll
    # provide a sha256 value for the default build
    sha256 = ""
    if download_url == "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.1.0.zip":
        sha256 = "c863a0073ff4c7b6feb958799c7dc3202b3449e86ff1cec9c85c7da9d1fe0218"

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
