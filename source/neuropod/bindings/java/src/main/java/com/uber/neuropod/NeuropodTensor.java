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

import java.io.Serializable;
import java.nio.*;
import java.util.List;

/**
 * This class is used for holding tensor data. It has its underlying C++ NeuropodTensor
 * object, should call close() function to free C++ side object when finish using the
 * NeuropodTensor object.
 * This object can be created by the Java side or by the C++ side(as an inference result).
 */
public class NeuropodTensor extends NativeClass implements Serializable {

    protected ByteBuffer buffer;

    private final boolean isFromJava;

    // Constructor for NeuropodTensor created in Java side.
    protected NeuropodTensor() {
        isFromJava = true;
    }

    // Constructor for NeuropodTensor created in C++ side.
    protected NeuropodTensor(long handle) {
        super(handle);
        isFromJava = false;
        buffer = nativeGetBuffer(handle).order(ByteOrder.nativeOrder());
    }

    /**
     * Get the dims array which represnents the shape of a tensor.
     *
     * @return the shape array
     */
    public long[] getDims() {
        return nativeGetDims(super.getNativeHandle());
    }

    /**
     * Gets the number of elements of a tensor.
     *
     * @return the number of elements
     */
    public long getNumberOfElements() {
        return nativeGetNumberOfElements(super.getNativeHandle());
    }

    /**
     * Gets the type of a tensor
     *
     * @return the tensor type
     */
    public TensorType getTensorType() {
        return nativeGetTensorType(super.getNativeHandle());
    }

    /**
     * Flatten the tensor data and convert it to a long buffer.
     * <p>
     * Can only be used when the tensor is INT64_TENSOR. Will trigger
     * a copy if the tensor is created by infer method. Otherwise will
     * not trigger a copy.
     *
     * @return the IntBuffer
     */
    public LongBuffer toLongBuffer() {
        checkType(TensorType.INT64_TENSOR);
        if (isFromJava) {
            return buffer.asLongBuffer();
        }
        LongBuffer ret = LongBuffer.allocate((int)getNumberOfElements()).put(buffer.asLongBuffer());
        ret.rewind();
        return ret;
    }

    /**
     * Flatten the tensor data and convert it to a float buffer.
     * <p>
     * Can only be used when the tensor is FLOAT_TENSOR. Will trigger
     * a copy if the tensor is created by infer method. Otherwise will
     * not trigger a copy.
     *
     * @return the FloatBuffer
     */
    public FloatBuffer toFloatBuffer() {
        checkType(TensorType.FLOAT_TENSOR);
        if (isFromJava) {
            return buffer.asFloatBuffer();
        }
        FloatBuffer ret = FloatBuffer.allocate((int)getNumberOfElements()).put(buffer.asFloatBuffer());
        ret.rewind();
        return ret;
    }

     /**
      * Flatten the tensor data and convert it to a int buffer.
      * <p>
      * Can only be used when the tensor is INT32_TENSOR. Will trigger
      * a copy.
      *
      * @return the IntBuffer
      */
     public IntBuffer toIntBuffer() {return null;}

     /**
      * Flatten the tensor data and convert it to a double buffer.
      * <p>
      * Can only be used when the tensor is DOUBLE_TENSOR. Will trigger
      * a copy.
      *
      * @return the DoubleBuffer
      */
     public DoubleBuffer toDoubleBuffer() {return null;}

     /**
      * Flatten the tensor data and convert it to a string list.
      * <p>
      * Can only be used when the tensor is STRING_TENSOR. Will trigger
      * a copy.
      *
      * @return the List
      */
     List<String> toStringList() {return null;}

     /**
      * Gets an int element at the given index.
      * <p>
      * Can only be used when the tensor is INT32_TENSOR.
      * For example, to get an int element from a 3rd-order tensor,
      * the user should call getInt(x, y, z)
      *
      * @param index the index array
      * @return the int element
      */
     int getInt(long... index) {return 0;}

     /**
      * Gets a long element at the given index.
      * <p>
      * Can only be used when the tensor is INT64_TENSOR.
      * For example, to get an long element from a 3rd-order tensor,
      * the user should call getLong(x, y, z)
      *
      * @param index the index array
      * @return the long element
      */
     long getLong(long... index) {return 0;}

     /**
      * Gets a double element at the given index.
      * <p>
      * Can only be used when the tensor is DOUBLE_TENSOR.
      * For example, to get a double element from a 3rd-order tensor,
      * the user should call getDouble(x, y, z)
      *
      * @param index the index array
      * @return the double element
      */
     double getDouble(long... index) {return 0.0;}

     /**
      * Gets a float element at the given index.
      * <p>
      * Can only be used when the tensor is FLOAT_TENSOR.
      * For example, to get a float element from a 3rd-order tensor,
      * the user should call getFloat(x, y, z)
      *
      * @param index the index array
      * @return the float element
      */
     float getFloat(long... index) {return 0.0f;}

     /**
      * Gets a string element at the given index.
      * <p>
      * Can only be used when the tensor is STRING_TENSOR.
      * For example, to get a string element from a 3rd-order tensor,
      * the user should call getString(x, y, z)
      *
      * @param index the index array
      * @return the string element
      */
     String getString(long... index) {return "";} // Not implemented yet

    private void checkType(TensorType type) {
        if (getTensorType() != type) {
            throw new NeuropodJNIException("TensorType mismatch! Expected " +
                    type.name() + ", found " + getTensorType().name());

        }
    }

    // Easier for the JNI side to call methods of super class.
    private long getHandle() {
        return super.getNativeHandle();
    }

    @Override
    protected void nativeDelete(long handle) throws NeuropodJNIException {
        nativeDoDelete(handle);
        this.buffer = null;
    }

    private static native void nativeDoDelete(long handle) throws NeuropodJNIException;

    private static native long[] nativeGetDims(long nativeHandle) throws NeuropodJNIException;

    private static native TensorType nativeGetTensorType(long nativeHandle) throws NeuropodJNIException;

    private static native long nativeGetNumberOfElements(long nativeHandle) throws NeuropodJNIException;

    private static native ByteBuffer nativeGetBuffer(long nativeHandle);
}
