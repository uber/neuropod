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
import java.util.Arrays;
import java.util.List;

/**
 * This class is used for holding tensor data. It has its underlying C++ NeuropodTensor
 * object, should call close() function to free C++ side object when finish using the
 * NeuropodTensor object.
 * This object can be created by the Java side or by the C++ side(as an inference result).
 */
public class NeuropodTensor extends NativeClass implements Serializable {

    protected ByteBuffer buffer;

    // This flag is used to separate NeuropodTensor that is created as Input (from java)
    // or Output (from JNI as result of inference). This is necessary because tensor buffer
    // is allocated differently and it is not safe to expose data directly to user.
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
     * Can only be used when the tensor is INT64_TENSOR.
     *
     * Note: It returns a buffer that is valid even after Tensor is closed.
     *
     * @return the IntBuffer
     */
    public LongBuffer toLongBuffer() {
        checkType(TensorType.INT64_TENSOR);
        if (isFromJava) {
            return buffer.asLongBuffer();
        }
        LongBuffer ret = LongBuffer.allocate((int) getNumberOfElements()).put(buffer.asLongBuffer());
        ret.rewind();
        return ret;
    }

    /**
     * Flatten the tensor data and convert it to a float buffer.
     * <p>
     * Can only be used when the tensor is FLOAT_TENSOR.
     *
     * Note: It returns a buffer that is valid even after Tensor is closed.
     *
     * @return the FloatBuffer
     */
    public FloatBuffer toFloatBuffer() {
        checkType(TensorType.FLOAT_TENSOR);
        if (isFromJava) {
            return buffer.asFloatBuffer();
        }
        FloatBuffer ret = FloatBuffer.allocate((int) getNumberOfElements()).put(buffer.asFloatBuffer());
        ret.rewind();
        return ret;
    }

    /**
     * Flatten the tensor data and convert it to a int buffer.
     * <p>
     * Can only be used when the tensor is INT32_TENSOR.
     *
     * Note: It returns a buffer that is valid even after Tensor is closed.
     *
     * @return the IntBuffer
     */
    public IntBuffer toIntBuffer() {
        checkType(TensorType.INT32_TENSOR);
        if (isFromJava) {
            return buffer.asIntBuffer();
        }
        IntBuffer ret = IntBuffer.allocate((int) getNumberOfElements()).put(buffer.asIntBuffer());
        ret.rewind();
        return ret;
    }

    /**
     * Flatten the tensor data and convert it to a double buffer.
     * <p>
     * Can only be used when the tensor is DOUBLE_TENSOR.
     *
     * Note: It returns a buffer that is valid even after Tensor is closed.
     *
     * @return the DoubleBuffer
     */
    public DoubleBuffer toDoubleBuffer() {
        checkType(TensorType.DOUBLE_TENSOR);
        if (isFromJava) {
            return buffer.asDoubleBuffer();
        }
        DoubleBuffer ret = DoubleBuffer.allocate((int) getNumberOfElements()).put(buffer.asDoubleBuffer());
        ret.rewind();
        return ret;
    }

    /**
     * Get the underlying raw memory data of the tensor.
     * <p>
     * WARNING: The returned buffer will be invalid once the tensor is closed.
     * It should not be called after the tensor has been closed.
     *
     * @return the ByteBuffer
     */

    public ByteBuffer getByteBuffer() {
        return buffer;
    }

    /**
     * Flatten the tensor data and convert it to a string list.
     * <p>
     * Can only be used when the tensor is STRING_TENSOR.
     *
     * Note: It returns list of Strings that is valid even after Tensor is closed.
     *
     * @return the List
     */
    public List<String> toStringList() {
        checkType(TensorType.STRING_TENSOR);
        return nativeToStringList(super.getNativeHandle());
    }

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
    public int getInt(long... index) {
        checkType(TensorType.INT32_TENSOR);
        long pos = toPos(index);
        return buffer.asIntBuffer().get((int) pos);
    }

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
    public long getLong(long... index) {
        checkType(TensorType.INT64_TENSOR);
        long pos = toPos(index);
        return buffer.asLongBuffer().get((int) pos);
    }

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
    public double getDouble(long... index) {
        checkType(TensorType.DOUBLE_TENSOR);
        long pos = toPos(index);
        return buffer.asDoubleBuffer().get((int) pos);
    }

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
    public float getFloat(long... index) {
        checkType(TensorType.FLOAT_TENSOR);
        long pos = toPos(index);
        return buffer.asFloatBuffer().get((int) pos);
    }

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
    public String getString(long... index) {
        checkType(TensorType.STRING_TENSOR);
        long pos = toPos(index);
        return nativeGetString(pos, super.getNativeHandle());
    }

    private void checkType(TensorType type) {
        if (getTensorType() != type) {
            throw new NeuropodJNIException("tensorType mismatch, expected " +
                    type.name() + ", found " + getTensorType().name());

        }
    }

    private long toPos(long[] index) {
        long[] dims = getDims();
        if (index.length != dims.length) {
            throw new java.lang.IndexOutOfBoundsException("trying to access index "
                    + Arrays.toString(index) + ", but the actual dims is " + Arrays.toString(dims));
        }

        long pos = 0;
        long acc = 1;
        for (int i = dims.length - 1; i >= 0; i--) {
            if (index[i] >= dims[i]) {
                throw new java.lang.IndexOutOfBoundsException("trying to access index "
                        + Arrays.toString(index) + ", but the actual dims is " + Arrays.toString(dims));
            }
            pos += index[i] * acc;
            if (i != 0) {
                acc *= dims[i];
            }
        }

        return pos;
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

    private static native List<String> nativeToStringList(long handle) throws NeuropodJNIException;

    private static native String nativeGetString(long pos, long modelHandle);
}
