/* Copyright (c) 2020 The Neuropod Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include <memory>
#include <utility>

namespace neuropod
{
namespace stdx
{
namespace detail
{

template <typename T>
struct unique_if
{
    using unique_ptr = std::unique_ptr<T>;
};

template <typename T>
struct unique_if<T[]>
{
    using unique_ptr_array_unknown_bound = std::unique_ptr<T[]>;
};

template <typename T, std::size_t N>
struct unique_if<T[N]>
{
    using unique_ptr_array_known_bound = void;
};

} // namespace detail

// See http://en.cppreference.com/w/cpp/memory/unique_ptr/make_unique
template <typename T, typename... Args>
typename detail::unique_if<T>::unique_ptr make_unique(Args &&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename T>
typename detail::unique_if<T>::unique_ptr_array_unknown_bound make_unique(std::size_t n)
{
    return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
}

template <typename T, typename... Args>
typename detail::unique_if<T>::unique_ptr_array_known_bound make_unique(Args &&...) = delete;

} // namespace stdx
} // namespace neuropod
