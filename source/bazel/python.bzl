# https://docs.bazel.build/versions/master/skylark/repository_rules.html
def _impl(repository_ctx):
    res = repository_ctx.execute(["python", "-c", "import os; from distutils import sysconfig; print(os.path.dirname(sysconfig.get_config_var('LIBDIR')))"])
    if res.return_code != 0:
        fail("Error getting python libdir: " + res.stderr)
    python_path = repository_ctx.path(res.stdout.strip('\n'))
    for f in python_path.readdir():
        repository_ctx.symlink(f, f.basename)

    # Find the python version
    res = repository_ctx.execute(["python", "-c", "import sys; print(sys.version[:3] + getattr(sys, 'abiflags', ''))"])
    if res.return_code != 0:
        fail("Error getting python version: " + res.stderr)
    python_version = res.stdout.strip('\n')

    # Generate a build file based on the template
    repository_ctx.template(
        "BUILD.bazel",
        repository_ctx.path(Label(repository_ctx.attr.build_file_template)),
        substitutions = {
            "{PYTHON_VERSION}": python_version,
        },
    )

python_repository = repository_rule(
    implementation=_impl,
    local=True,
    attrs={
        "build_file_template": attr.string(mandatory=True),
    }
)
