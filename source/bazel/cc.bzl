# Build rules that let us pass in a set of warning flags for some targets
# without requiring us to pass them to all targets (e.g. third party libraries)

_warnings = [
    "-Weverything",
    "-Wno-c++98-compat",
    "-Wno-c++98-compat-pedantic",
    "-Wno-padded",

    # This is a warning that a move would not have been applied on old compilers
    # https://reviews.llvm.org/D43322
    "-Wno-return-std-move-in-c++11",

    # We need this for polymorphism with header only implementations
    "-Wno-weak-vtables",

    # We're okay with these for now, but should refactor to remove if we can
    "-Wno-global-constructors",
    "-Wno-exit-time-destructors",

    # It's okay if we use `default` in switch statements
    "-Wno-switch-enum",

    # Flexible array members
    "-Wno-c99-extensions",
]

_test_warnings = _warnings + [
    # Test specific warnings and exceptions
]

def neuropod_cc_library(name, copts = [], **kwargs):
    native.cc_library(
        name = name,
        copts = _warnings + copts,
        **kwargs
    )

def neuropod_cc_binary(name, copts = [], **kwargs):
    native.cc_binary(
        name = name,
        copts = _warnings + copts,
        **kwargs
    )

def neuropod_cc_test(name, copts = [], **kwargs):
    native.cc_test(
        name = name,
        copts = _test_warnings + copts,
        **kwargs
    )
