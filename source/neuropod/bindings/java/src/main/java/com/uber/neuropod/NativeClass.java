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

package com.uber.neuropod;

/**
 * This is a base class for all class with a binding to native class.
 * Need to call close() method after usage to free memory in C++ side.
 * This class is not thread-safe.
 */
abstract class NativeClass implements AutoCloseable {
    // Load native library
    static {
        LibraryLoader.load();
    }

    // The pointer to the native object
    private long nativeHandle;

    /**
     * Instantiates a new Native class.
     */
    public NativeClass() {
    }

    /**
     * Instantiates a new Native class.
     *
     * @param handle the native handle
     */
    protected NativeClass(long handle) {
        nativeHandle = handle;
    }

    /**
     * Gets native handle.
     *
     * @return the native handle
     */
    protected final long getNativeHandle() {
        if (nativeHandle == 0) {
            throw new NeuropodJNIException("deallocated object");
        }
        return nativeHandle;
    }

    /**
     * Sets native handle.
     *
     * @param handle the handle
     */
    protected final void setNativeHandle(long handle) {
        this.nativeHandle = handle;
    }

    /**
     * Delete the underlying native class
     *
     * @param handle the handle
     */
    abstract protected void nativeDelete(long handle) throws NeuropodJNIException;

    @Override
    public final void close() throws NeuropodJNIException {
        // Wrap the nativeDelete to close method so that the IDE will have a warning
        // if the object is not deleted, and will be auto deleted in a try catch block.
        if (nativeHandle == 0) {
            return;
        }
        this.nativeDelete(nativeHandle);
        this.nativeHandle = 0;
    }
}
