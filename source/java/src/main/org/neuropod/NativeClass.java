package org.neuropod;

/**
 * This is a base class for all class with a binding to native class
 */
public abstract class NativeClass implements AutoCloseable {

    // The pointer to the native object
    private long nativeHandle_;

    /**
     * Instantiates a new Native class.
     */
    public NativeClass() {}

    /**
     * Instantiates a new Native class.
     *
     * @param nativeHandle the native handle
     */
    public NativeClass(long nativeHandle) {
        this.nativeHandle_ = nativeHandle;
    }

    /**
     * Gets native handle.
     *
     * @return the native handle
     */
    protected long getNativeHandle() {
        return nativeHandle_;
    }

    /**
     * Sets native handle.
     *
     * @param handle the handle
     */
    protected void setNativeHandle(long handle) {
        nativeHandle_ = handle;
    }

    /**
     * Delete the underlying native class
     *
     * @param handle the handle
     */
    abstract protected void nativeDelete(long handle);

    @Override
    public void close() throws Exception {
        // Wrap the nativeDelete to close method so that the IDE will have a warning
        // if the object is not deleted, and will be auto deleted in a try catch block.
        nativeDelete(nativeHandle_);
        nativeHandle_ = 0;
    }
}
