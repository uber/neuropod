#
# Uber, Inc. (c) 2018
#

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "python-numpy_staticlib",
    srcs = ["numpy/core/lib/libnpymath.a"],
)

filegroup(
    name = "python-numpy_otherlib",
    srcs = [
        "numpy/core/_dotblas.so",
        "numpy/core/_dummy.so",
        "numpy/core/multiarray.so",
        "numpy/core/multiarray_tests.so",
        "numpy/core/operand_flag_tests.so",
        "numpy/core/scalarmath.so",
        "numpy/core/struct_ufunc_test.so",
        "numpy/core/test_rational.so",
        "numpy/core/umath.so",
        "numpy/core/umath_tests.so",
        "numpy/fft/fftpack_lite.so",
        "numpy/lib/_compiled_base.so",
        "numpy/linalg/_umath_linalg.so",
        "numpy/linalg/lapack_lite.so",
        "numpy/numarray/_capi.so",
        "numpy/random/mtrand.so",
    ],
)

cc_library(
    name = "python-numpy_hdrs",
    hdrs = glob([
        "numpy/core/include/numpy/*.h",
        "numpy/numarray/include/numpy/*.h",
    ]),
    includes = [
        "numpy/core/include",
        "numpy/numarray/include",
    ],
)

cc_library(
    name = "python_numpy",
    data = [
        ":python-numpy_otherlib",
        ":python-numpy_staticlib",
    ],
    deps = [":python-numpy_hdrs"],
)
