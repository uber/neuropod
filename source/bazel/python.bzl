# https://docs.bazel.build/versions/master/skylark/repository_rules.html
def _impl(repository_ctx):
    # The `or` pattern below handles empty strings and unset env variables
    # Using a default value only handles unset env variables
    version = repository_ctx.os.environ.get("NEUROPOD_PYTHON_VERSION") or "2.7"
    IS_MAC = repository_ctx.os.name.startswith("mac")

    if IS_MAC:
        # Get the libdir
        res = repository_ctx.execute(["python", "-c", "import os; from distutils import sysconfig; print(os.path.dirname(sysconfig.get_config_var('LIBDIR')))"])
        if res.return_code != 0:
            fail("Error getting python libdir: " + res.stderr)

        # Create a symlink
        python_path = repository_ctx.path(res.stdout.strip("\n"))
        for f in python_path.readdir():
            repository_ctx.symlink(f, f.basename)

    else:
        # TODO(vip): Add mac binaries
        MAPPING = {
            # Linux
            "2.7-linux": {
                "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/v0.0.1/python-2.7.17.tar.gz",
                "sha256": "8edb75fb76873ae2eba21ef5c677cf29864b33f6abbf3928d010baab28dcc67e",
            },
            "3.5-linux": {
                "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/v0.0.1/python-3.5.9.tar.gz",
                "sha256": "d5b83a4565ccd746ce312fcca9998c2100aee37db807d37f42ff43c17e9f5dd7",
            },
            "3.6-linux": {
                "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/v0.0.1/python-3.6.10.tar.gz",
                "sha256": "f8d2e7b5468464ed653f832b363ebf228108ecc1744f0915cdbed2ab31eda99a",
            },
            "3.7-linux": {
                "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/v0.0.1/python-3.7.7.tar.gz",
                "sha256": "53eb870e33b7581b44f95f79fdbeb275ab3a03794270d3f5cb64699d7c65e2fa",
            },
            "3.8-linux": {
                "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/v0.0.1/python-3.8.2.tar.gz",
                "sha256": "8a93f738894db779c282a02fb7a88e4911538e26ed834a23bb1bc9f3e2fe9e04",
            },
        }

        download_mapping = MAPPING["{}-{}".format(
            version,
            "mac" if IS_MAC else "linux",
        )]

        download_url = download_mapping["url"]
        sha256 = download_mapping["sha256"]

        repository_ctx.download_and_extract(download_url, sha256 = sha256)

    # Generate a build file based on the template
    repository_ctx.template(
        "BUILD.bazel",
        repository_ctx.path(Label(repository_ctx.attr.build_file_template)),
        substitutions = {
            "{PYTHON_VERSION}": version,
        },
    )

python_repository = repository_rule(
    implementation = _impl,
    local = True,
    attrs = {
        "build_file_template": attr.string(mandatory = True),
    },
)
