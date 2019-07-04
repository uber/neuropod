# https://docs.bazel.build/versions/master/skylark/repository_rules.html
def _impl(repository_ctx):
    # The `or` pattern below handles empty strings and unset env variables
    # Using a default value only handles unset env variables
    version = repository_ctx.os.environ.get("NEUROPODS_TENSORFLOW_VERSION") or "1.12.0"
    IS_MAC  = repository_ctx.os.name.startswith("mac")
    IS_GPU  = (repository_ctx.os.environ.get("NEUROPODS_IS_GPU") or None) != None

    # TODO(vip): find libtensorflow nightly builds
    download_url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-"

    if IS_GPU:
        download_url += "gpu"
    else:
        download_url += "cpu"

    if IS_MAC:
        download_url += "-darwin-x86_64-" + version + ".tar.gz"
    else:
        download_url += "-linux-x86_64-" + version + ".tar.gz"

    repository_ctx.download_and_extract(download_url)
    repository_ctx.symlink(repository_ctx.path(Label(repository_ctx.attr.build_file)), "BUILD.bazel")

tensorflow_repository = repository_rule(
    implementation=_impl,
    local=True,
    attrs={"build_file": attr.string(mandatory=True)})
