package com.uber.neuropod;

/**
 * All exceptions triggered in the C++ side of Neuropod
 */
public class NeuropodJNIException extends RuntimeException {
    /**
     * Instantiates a new Neuropod jni exception.
     *
     * @param message the message
     */
    public NeuropodJNIException(String message) {
        super(message);
    }
}
