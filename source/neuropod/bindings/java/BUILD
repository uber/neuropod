# Copyright (c) 2020 UATC, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(":java_build_defs.bzl", "JAVACOPTS")

genrule(
    name = "copy_link_jni_md_header",
    srcs = select({
        "@bazel_tools//src/conditions:darwin": ["@bazel_tools//tools/jdk:jni_md_header-darwin"],
        "//conditions:default": ["@bazel_tools//tools/jdk:jni_md_header-linux"],
    }),
    outs = ["jni_md.h"],
    cmd = "cp -f $< $@",
)

genrule(
    name = "copy_link_jni_header",
    srcs = ["@bazel_tools//tools/jdk:jni_header"],
    outs = ["jni.h"],
    cmd = "cp -f $< $@",
)

filegroup(
    name = "backends",
    srcs = [
        "//neuropod/backends/python_bridge:libneuropod_pythonbridge_backend.so",
        "//neuropod/backends/tensorflow:libneuropod_tensorflow_backend.so",
        "//neuropod/backends/torchscript:libneuropod_torchscript_backend.so",
        "@libtorch_repo//:libtorch_libs",
        "@tensorflow_repo//:libtensorflow_libs",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
)

filegroup(
    name = "tar_backends",
    srcs = [
        "//neuropod/backends/tensorflow:neuropod_tensorflow_backend",
        "//neuropod/backends/torchscript:neuropod_torchscript_backend",
    ],
    visibility = [
        "//neuropod:__subpackages__",
    ],
)

# This genrule copies to a dylib and changes the libneuropod.so dependency path
genrule(
    name = "copy_to_dylib",
    srcs = [":libneuropod_jni.so"],
    outs = ["libneuropod_jni.dylib"],
    cmd = "cp -f $< $@",
)

filegroup(
    name = "neuropod_jni",
    srcs = select({
        "@bazel_tools//src/conditions:darwin": [":copy_to_dylib"],
        "//conditions:default": [":libneuropod_jni.so"],
    }),
    visibility = [
        "//neuropod:__subpackages__",
    ],
)

# This genrule changes the libneuropod.so dependency path for darwin platform
genrule(
    name = "updated_neuropod_multiprocess_worker",
    srcs = ["//neuropod/multiprocess:neuropod_multiprocess_worker"],
    outs = ["neuropod_multiprocess_worker"],
    cmd = select({
        "@bazel_tools//src/conditions:darwin": "cp -f $< $@; chmod 755 $@; install_name_tool -change \"$$(otool -L $@ | grep libneuropod.so | cut -d ' ' -f1 | column -t)\" \"@rpath/libneuropod.so\" \"$@\"",
        "//conditions:default": "cp -f $< $@",
    }),
)

filegroup(
    name = "java_source",
    srcs = glob(["src/main/java/com/uber/neuropod/*.java"]),
    visibility = [
        "//neuropod:__subpackages__",
    ],
)

# Unpack jar file in tmp directory, move files to expected subdirectories, pack to jar file again.
# Note it runs inside sandbox where all created files and dirs are temporarily that deleted after build.
genrule(
    name = "neuropod_java_jar",
    srcs = [":neuropod_java_tmp.jar"],
    outs = ["neuropod_java_jar.jar"],
    cmd = select({
        "@bazel_tools//src/conditions:darwin": """
mkdir tmp
cd tmp
jar -xf ../$(location :neuropod_java_tmp.jar)
jni_path="./com/uber/neuropod/native/darwin-x86_64"
mkdir -p $$jni_path
mv libneuropod_jni.dylib libneuropod.so neuropod_multiprocess_worker $$jni_path
mv neuropod_tensorflow_backend.tar.gz $$jni_path
mv neuropod_torchscript_backend.tar.gz $$jni_path
cd ..
jar cf $@ -C tmp/ .  # Creates the new jar file.
""",
        "//conditions:default": """
mkdir tmp
cd tmp
jar -xf ../$(location :neuropod_java_tmp.jar)
jni_path="./com/uber/neuropod/native/linux-x86_64"
mkdir -p $$jni_path
mv libneuropod_jni.so libneuropod.so neuropod_multiprocess_worker $$jni_path
mv neuropod_tensorflow_backend.tar.gz $$jni_path
mv neuropod_torchscript_backend.tar.gz $$jni_path
cd ..
jar cf $@ -C tmp/ .  # Creates the new jar file.
""",
    }),
)

java_binary(
    name = "neuropod_java_tmp",
    srcs = [":java_source"],
    # Need a way to make these native libraries locate at com/uber/neuropod/native/OS-ARCH in the jar file.
    # Bazel doesn't provide official way for that, use genrule neuropod_java_jar workaround for now.
    classpath_resources = [
        "//neuropod:libneuropod.so",
        ":updated_neuropod_multiprocess_worker",
        ":neuropod_jni",
        ":tar_backends",
    ],
    javacopts = JAVACOPTS,
    visibility = ["//visibility:public"],
)

java_library(
    name = "neuropod_java",
    srcs = [":java_source"],
    data = [
        ":backends",
        ":neuropod_jni",
        ":updated_neuropod_multiprocess_worker",
        "//neuropod:libneuropod.so",
    ],
    javacopts = JAVACOPTS,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "copy_jni_hdr_lib",
    hdrs = [
        ":copy_link_jni_header",
        ":copy_link_jni_md_header",
    ],
    includes = ["."],
)

cc_binary(
    name = "libneuropod_jni.so",
    srcs = glob(["src/main/native/*.cc"]) + glob(["src/main/native/*.h"]) + [
        "//neuropod:libneuropod.so",
    ],
    includes = select({
        "@bazel_tools//src/conditions:darwin": ["external/local_jdk/include/darwin"],
        "//conditions:default": ["external/local_jdk/include/linux"],
    }),
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": ["-Wl,-rpath,@loader_path"],
        "//conditions:default": [
            "-Wl,-rpath,$$ORIGIN",
        ],
    }),
    linkshared = True,
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":copy_jni_hdr_lib",
        "//neuropod:neuropod_hdrs",
    ],
)

java_test(
    name = "LibraryLoaderTest",
    srcs = ["src/test/java/com/uber/neuropod/LibraryLoaderTest.java"],
    javacopts = JAVACOPTS,
    test_class = "com.uber.neuropod.LibraryLoaderTest",
    deps = [
        ":neuropod_java_jar",
        "@junit",
    ],
)

java_test(
    name = "TFAdditionTest",
    srcs = [
        "src/test/java/com/uber/neuropod/NeuropodAdditionTest.java",
        "src/test/java/com/uber/neuropod/TFAdditionTest.java",
    ],
    data = [
        "//neuropod/tests/test_data",
    ],
    javacopts = JAVACOPTS,
    test_class = "com.uber.neuropod.TFAdditionTest",
    deps = [
        ":neuropod_java_jar",
        "@junit",
    ],
)

java_test(
    name = "TorchscriptAdditionTest",
    srcs = [
        "src/test/java/com/uber/neuropod/NeuropodAdditionTest.java",
        "src/test/java/com/uber/neuropod/TorchscriptAdditionTest.java",
    ],
    data = [
        "//neuropod/tests/test_data",
    ],
    javacopts = JAVACOPTS,
    test_class = "com.uber.neuropod.TorchscriptAdditionTest",
    deps = [
        ":neuropod_java_jar",
        "@junit",
    ],
)

java_test(
    name = "TensorSpecTest",
    srcs = ["src/test/java/com/uber/neuropod/TensorSpecTest.java"],
    data = [
        "//neuropod/tests/test_data",
    ],
    javacopts = JAVACOPTS,
    test_class = "com.uber.neuropod.TensorSpecTest",
    deps = [
        ":neuropod_java_jar",
        "@junit",
    ],
)

java_test(
    name = "NeuropodTensorAllocatorTest",
    srcs = ["src/test/java/com/uber/neuropod/NeuropodTensorAllocatorTest.java"],
    data = [
        "//neuropod/tests/test_data",
    ],
    javacopts = JAVACOPTS,
    # Tagged as GPU so we have at least one Java GPU test. This helps keep CI scripts simple
    # TODO(vip): Remove once we have a Java test that actually requires GPU
    tags = ["gpu"],
    test_class = "com.uber.neuropod.NeuropodTensorAllocatorTest",
    deps = [
        ":neuropod_java_jar",
        "@junit",
    ],
)
