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

import java.nio.*;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * The factory type for NeuropodTensor, can be obtained from a neuropod model or as a generic allocator.
 * Should call close method when finish using it.
 */
public class NeuropodTensorAllocator extends NativeClass {
    protected NeuropodTensorAllocator(long handle) {
        super(handle);
    }

    private static final Set<TensorType> SUPPORTED_TENSOR_TYPES = new HashSet<>(Arrays.asList(
            TensorType.INT32_TENSOR,
            TensorType.INT64_TENSOR,
            TensorType.DOUBLE_TENSOR,
            TensorType.FLOAT_TENSOR));

    /**
     * Create a NeuropodTensor based on given ByteBuffer, dims array and tensorType. The created tensor
     * will have the input tensorType.
     * <p>
     * For the sake of performance we require that the input buffer is direct, native and not read-only
     * because in this case Java virtual machine will perform native I/O operations directly upon it.
     * A direct byte buffer may be created by invoking the allocateDirect factory method of this class.
     * Will not trigger a copy.
     *
     * @param byteBuffer the buffer that contains tensor data
     * @param dims       the shape of the tensor
     * @param tensorType the tensor type
     * @return the created NeuropodTensor
     */
    public NeuropodTensor tensorFromMemory(ByteBuffer byteBuffer, long[] dims, TensorType tensorType) {
        if (!SUPPORTED_TENSOR_TYPES.contains(tensorType)) {
            throw new NeuropodJNIException("unsupported tensor type: " + tensorType.name());
        }
        if (!byteBuffer.isDirect()) {
            throw new NeuropodJNIException("the input byteBuffer is not direct");
        }
        if (byteBuffer.order() != ByteOrder.nativeOrder()) {
            throw new NeuropodJNIException("the input byteBuffer is not in a native order");
        }
        if (byteBuffer.isReadOnly()) {
            throw new NeuropodJNIException("the input byteBuffer is read-only");
        }
        return createTensorFromBuffer(byteBuffer, dims, tensorType);
    }

    /**
     * Create a NeuropodTensor based on given LongBuffer, dims array. The created tensor
     * will have tensor type INT64_TENSOR.
     * <p>
     * Will not trigger a copy if the buffer is a direct buffer in native order and the backend of
     * the allocator does not have alignment requirement. Otherwise it will trigger a copy.
     *
     * @param longBuffer the buffer that contains tensor data
     * @param dims       the shape of the tensor
     * @return the created NeuropodTensor
     */
    NeuropodTensor create(LongBuffer longBuffer, long[] dims) {
        return null;
    }

    /**
     * Create a NeuropodTensor based on given IntBuffer, dims array. The created tensor
     * will have type INT32_TENSOR.
     * <p>
     * Will not trigger a copy if the buffer is a direct buffer in native order and the backend of
     * the allocator does not have alignment requirement. Otherwise it will trigger a copy.
     *
     * @param intBuffer the buffer that contains tensor data
     * @param dims      the shape of the tensor
     * @return the created NeuropodTensor
     */
    NeuropodTensor create(IntBuffer intBuffer, long[] dims) {
        return null;
    }

    /**
     * Create a NeuropodTensor based on given DoubleBuffer, dims array. The created tensor
     * will have type DOUBLE_TENSOR.
     * <p>
     * Will not trigger a copy if the buffer is a direct buffer in native order and the backend of
     * the allocator does not have alignment requirement. Otherwise it will trigger a copy.
     *
     * @param doubleBuffer the buffer that contains tensor data
     * @param dims         the shape of the tensor
     * @return the created NeuropodTensor
     */
    NeuropodTensor create(DoubleBuffer doubleBuffer, long[] dims) {
        return null;
    }

    /**
     * Create a NeuropodTensor based on given FloatBuffer, dims array. The created tensor
     * will have type FLOAT_TENSOR.
     * <p>
     * Will not trigger a copy if the buffer is a direct buffer in native order and the backend of
     * the allocator does not have alignment requirement. Otherwise it will trigger a copy.
     *
     * @param floatBuffer the buffer that contains tensor data
     * @param dims        the shape of the tensor
     * @return the neuropod NeuropodTensor
     */
    NeuropodTensor create(FloatBuffer floatBuffer, long[] dims) {
        return null;
    }

    /**
     * Create a NeuropodTensor based on given string list, dims array. The created tensor
     * will have type STRING_TENSOR.
     * <p>
     * Will trigger a copy.
     *
     * @param stringList the string list
     * @param dims       the dims
     * @return the created NeuropodTensor
     */
    NeuropodTensor create(List<String> stringList, long[] dims) {
        return null;
    }

    private NeuropodTensor createTensorFromBuffer(ByteBuffer buffer, long[] dims, TensorType tensorType) {
        NeuropodTensor tensor = new NeuropodTensor();
        tensor.buffer = buffer;
        tensor.setNativeHandle(nativeAllocate(dims, tensorType.getValue(), tensor.buffer, super.getNativeHandle()));
        return tensor;
    }

    private static native long nativeAllocate(long[] dims,
                                              int tensorType,
                                              ByteBuffer buffer,
                                              long allocatorHandle) throws NeuropodJNIException;

    @Override
    protected native void nativeDelete(long handle) throws NeuropodJNIException;
}
