# https://docs.bazel.build/versions/master/skylark/repository_rules.html
def _impl(repository_ctx):
    res = repository_ctx.execute(["python", "-c", "import sys; print(sys.exec_prefix)"])
    if res.return_code != 0:
        fail("Python interpreter not found: " + res.stderr)
    python_path = repository_ctx.path(res.stdout.strip('\n'))
    for f in python_path.readdir():
        repository_ctx.symlink(f, f.basename)
    repository_ctx.symlink(repository_ctx.path(Label(repository_ctx.attr.build_file)), "BUILD.bazel")

python_repository = repository_rule(
    implementation=_impl,
    local=True,
    attrs={"build_file": attr.string(mandatory=True)})
