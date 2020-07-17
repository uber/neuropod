package com.uber.neuropod;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

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
