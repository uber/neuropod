# https://docs.bazel.build/versions/master/skylark/repository_rules.html
def _impl(repository_ctx):
    if repository_ctx.os.environ.get("NEUROPODS_LIBTORCH_URL"):
        download_url = repository_ctx.os.environ["NEUROPODS_LIBTORCH_URL"]
        download_sha = repository_ctx.os.environ.get("NEUROPODS_LIBTORCH_SHA256", '')
    else:
        version = repository_ctx.os.environ.get("NEUROPODS_LIBTORCH_VERSION", "") or repository_ctx.attr.default_version
        download_base = "https://download.pytorch.org/libtorch"
        if "dev" in version:
            download_base += "/nightly"

        if repository_ctx.os.name.startswith("mac"):
            download_url = download_base + "/cpu/libtorch-macos-" + version + ".zip"
        else:
            download_url = download_base + "/cpu/libtorch-shared-with-deps-" + version + ".zip"
        download_sha = ''

        if repository_ctx.attr.default_version == version:
            # If we're using the default version, use the default sha
            if repository_ctx.os.name.startswith("mac"):
                download_sha = repository_ctx.attr.mac_sha
            else:
                download_sha = repository_ctx.attr.linux_sha



    repository_ctx.download_and_extract(download_url, sha256=download_sha, stripPrefix="libtorch")
    repository_ctx.symlink(repository_ctx.path(Label(repository_ctx.attr.build_file)), "BUILD.bazel")

libtorch_repository = repository_rule(
    implementation=_impl,
    local=True,
    attrs={"build_file": attr.string(mandatory=True),
           "default_version": attr.string(mandatory=True),
           "linux_sha": attr.string(),
           "mac_sha": attr.string()})
