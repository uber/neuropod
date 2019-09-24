//
// Uber, Inc. (c) 2019
//

#pragma once

#include <caffe2/core/macros.h>
#include <torch/script.h>
#include <torch/torch.h>

namespace neuropods
{

// If we're not building with a nightly relase of torch,
// set the date to match the date of the official release
#ifndef CAFFE2_NIGHTLY_VERSION
#if CAFFE2_VERSION == 10200
// The date of the official torch 1.2.0 release
#define CAFFE2_NIGHTLY_VERSION 20190808
#endif
#endif

#if CAFFE2_NIGHTLY_VERSION >= 20190717
#define MAKE_DICT(name) c10::impl::GenericDict name((c10::impl::deprecatedUntypedDict()));
#elif CAFFE2_NIGHTLY_VERSION >= 20190601
#define MAKE_DICT(name) auto name = c10::make_dict<torch::jit::IValue, torch::jit::IValue>();
#else
#define MAKE_DICT(name) torch::ivalue::UnorderedMap name;
#endif

#if CAFFE2_NIGHTLY_VERSION >= 20190717
#define SCHEMA(method) method.function().getSchema()
#else
#define SCHEMA(method) method.getSchema()
#endif

#if CAFFE2_NIGHTLY_VERSION >= 20190601
#define KEY(elem) (elem.key())
#define VALUE(elem) (elem.value())
#define DICT_INSERT(dict, key, value) dict.insert(key, value);
#else
#define KEY(elem) (elem.first)
#define VALUE(elem) (elem.second)
#define DICT_INSERT(dict, key, value) dict[key] = value;
#endif

} // namespace neuropods
