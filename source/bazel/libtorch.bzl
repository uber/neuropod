# https://docs.bazel.build/versions/master/skylark/repository_rules.html
def _impl(repository_ctx):
    # The `or` pattern below handles empty strings and unset env variables
    # Using a default value only handles unset env variables
    version = repository_ctx.os.environ.get("NEUROPODS_TORCH_VERSION") or "1.1.0"
    IS_MAC  = repository_ctx.os.name.startswith("mac")
    IS_GPU  = (repository_ctx.os.environ.get("NEUROPODS_IS_GPU") or None) != None
    CUDA_VERSION = repository_ctx.os.environ.get("NEUROPODS_CUDA_VERSION") or "10.0"

    download_url = "https://download.pytorch.org/libtorch"
    if "dev" in version:
        download_url += "/nightly"

    if IS_GPU:
        download_url += "/cu" + CUDA_VERSION.replace(".", "")
    else:
        download_url += "/cpu"

    if IS_MAC:
        download_url += "/libtorch-macos-" + version + ".zip"
    else:
        download_url += "/libtorch-shared-with-deps-" + version + ".zip"

    repository_ctx.download_and_extract(download_url, stripPrefix="libtorch")
    repository_ctx.symlink(repository_ctx.path(Label(repository_ctx.attr.build_file)), "BUILD.bazel")

libtorch_repository = repository_rule(
    implementation=_impl,
    local=True,
    attrs={"build_file": attr.string(mandatory=True)})
