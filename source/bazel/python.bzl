# https://docs.bazel.build/versions/master/skylark/repository_rules.html
def _impl(repository_ctx):
    # The `or` pattern below handles empty strings and unset env variables
    # Using a default value only handles unset env variables
    version = repository_ctx.os.environ.get("NEUROPOD_PYTHON_VERSION") or "2.7"
    IS_MAC = repository_ctx.os.name.startswith("mac")

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
            "sha256": "be0917eadbe3dab44ed7370a20158148e37a270c77467ee888f7095e5ac79bfc",
        },

        # Mac
        "2.7-mac": {
            "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/v0.0.1/python-2.7.18-macosx10.9.tar.gz",
            "sha256": "830d79bd33cf184ecdd44026353cbc46af4460b857ac2045a0a84d29562b9775",
        },
        "3.5-mac": {
            "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/v0.0.1/python-3.5.4-macosx10.6.tar.gz",
            "sha256": "4547090a248dc98883093353926c386364156388530e5df8eca186ce23ba6c27",
        },
        "3.6-mac": {
            "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/v0.0.1/python-3.6.8-macosx10.9.tar.gz",
            "sha256": "117daaeed6a7cbfe66a661750ded078fc8945e98c813e9760e6fc29ccad7b3f4",
        },
        "3.7-mac": {
            "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/v0.0.1/python-3.7.7-macosx10.9.tar.gz",
            "sha256": "00127d7c2443932e1dcbd40518e8b89d35ef8998a1465afe5e37241e9f84646c",
        },
        "3.8-mac": {
            "url": "https://github.com/VivekPanyam/python-prebuilts/releases/download/v0.0.1/python-3.8.2-macosx10.9.tar.gz",
            "sha256": "9ba50ebc5729919f9051678a5b698dda12a4df6cf2d871bbb88d0581a3d43ae4",
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
    environ = [
        "NEUROPOD_PYTHON_VERSION",
    ],
    attrs = {
        "build_file_template": attr.string(mandatory = True),
    },
)
