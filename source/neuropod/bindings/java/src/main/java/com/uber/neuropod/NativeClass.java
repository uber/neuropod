/* Copyright (c) 2020 UATC, LLC

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

package com.uber.neuropod;

/**
 * This is a base class for all class with a binding to native class.
 * Need to call close() method after usage to free memory in C++ side.
 */
abstract class NativeClass implements AutoCloseable {
    // Load native library
    static {
        LibraryLoader.load();
    }

    @Override
    public void close() throws Exception {
        // Wrap the nativeDelete to close method so that we can do some common post-process here if needed.
    }
}
