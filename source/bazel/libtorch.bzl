# https://docs.bazel.build/versions/master/skylark/repository_rules.html
def _impl(repository_ctx):
    if repository_ctx.os.environ.get("NEUROPODS_PYTORCH_URL"):
        download_url = repository_ctx.os.environ["NEUROPODS_PYTORCH_URL"]
        download_sha = repository_ctx.os.environ.get("NEUROPODS_PYTORCH_SHA256", '')
    else:
        version = repository_ctx.os.environ.get("NEUROPODS_PYTORCH_VERSION", "") or repository_ctx.attr.default_version
        if repository_ctx.os.name.startswith("mac"):
            download_url = "https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-" + version + ".zip"
        else:
            download_url = "https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-" + version + ".zip"
        download_sha = ''

    repository_ctx.download_and_extract(download_url, sha256=download_sha, stripPrefix="libtorch")
    repository_ctx.symlink(repository_ctx.path(Label(repository_ctx.attr.build_file)), "BUILD.bazel")

libtorch_repository = repository_rule(
    implementation=_impl,
    local=True,
    attrs={"build_file": attr.string(mandatory=True),
           "default_version": attr.string(mandatory=True)})
