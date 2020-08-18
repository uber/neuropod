load("//bazel:version.bzl", "NEUROPOD_VERSION")

# https://docs.bazel.build/versions/master/skylark/repository_rules.html
def _impl(repository_ctx):
    # The `or` pattern below handles empty strings and unset env variables
    # Using a default value only handles unset env variables
    version = repository_ctx.os.environ.get("NEUROPOD_PYTHON_VERSION") or "2.7"
    IS_MAC = repository_ctx.os.name.startswith("mac")

    MAPPING = {
        # Linux
        "2.7-linux": {
            "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/0.0.2test/cp27-cp27mu.tar.gz",
            "sha256": "bc488d261fb9da9168ada77358331efa75fb7d84f5d2c4618c9c12ea8da821bb",
        },
        "3.5-linux": {
            "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/0.0.2test/cp35-cp35m.tar.gz",
            "sha256": "",
        },
        "3.6-linux": {
            "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/0.0.2test/cp36-cp36m.tar.gz",
            "sha256": "",
        },
        "3.7-linux": {
            "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/0.0.2test/cp37-cp37m.tar.gz",
            "sha256": "",
        },
        "3.8-linux": {
            "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/0.0.2test/cp38-cp38.tar.gz",
            "sha256": "",
        },

        # Mac
        "2.7-mac": {
            "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/0.0.2test/darwin_2.7.18_10.9.tar.gz",
            "sha256": "da903cc36026ae4644c9aeee044e3ce336b13975040cb796d9d4847e511dba9b",
        },
        "3.5-mac": {
            "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/0.0.2test/darwin_3.5.4_10.6.tar.gz",
            "sha256": "",
        },
        "3.6-mac": {
            "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/0.0.2test/darwin_3.6.8_10.9.tar.gz",
            "sha256": "",
        },
        "3.7-mac": {
            "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/0.0.2test/darwin_3.7.8_10.9.tar.gz",
            "sha256": "",
        },
        "3.8-mac": {
            "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/0.0.2test/darwin_3.8.5_10.9.tar.gz",
            "sha256": "",
        },
    }

    download_mapping = MAPPING["{}-{}".format(
        version,
        "mac" if IS_MAC else "linux",
    )]

    download_url = download_mapping["url"]
    sha256 = download_mapping["sha256"]

    repository_ctx.download(download_url, output = "python.tar.gz", sha256 = sha256)
    repository_ctx.extract("python.tar.gz")

    # Generate a build file based on the template
    repository_ctx.template(
        "BUILD.bazel",
        repository_ctx.path(Label(repository_ctx.attr.build_file_template)),
        substitutions = {
            "{PYTHON_VERSION}": version,
        },
    )

    # Create a file that specifies versioning information
    repository_ctx.file(
        "neuropod_backend_path.bzl",
        content = """
        NEUROPOD_BACKEND_PATH = "{}/backends/python_{}/"
        """.format(
            NEUROPOD_VERSION,
            version,
        ).strip(),
    )

python_repository = repository_rule(
    implementation = _impl,
    environ = [
        "NEUROPOD_PYTHON_VERSION",
    ],
    attrs = {
        "build_file_template": attr.string(mandatory = True),
    },
)
