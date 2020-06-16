package org.neuropod;

import java.util.List;

/**
 * This class is an one to one mapping to cpp NeuropodValue type.
 * Should not call the close function if this class does not have the ownership of the underlying native class.
 */
public class NeuropodValue extends NativeClass {
    /**
     * Instantiates a new Neuropod value map from existing cpp handle.
     *
     * @param nativeHandle the native handle
     */
    public NeuropodValue(long nativeHandle) {
        super(nativeHandle);
    }

    /**
     * Copy all the data of the NeuropodValue and flatten it to a java list.
     *
     * @return the list
     */
    public List<Object> toList() {
        return nativeToList(super.getNativeHandle());
    }

    static native private List<Object> nativeToList(long handle) throws NeuropodJNIException;

    @Override
    protected native void nativeDelete(long handle);
}
