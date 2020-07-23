package com.uber.neuropod;

import java.nio.*;
import java.util.List;

/**
 * The factory type for NeuropodTensor, can be obtained from a neuropod model or as a generic allocator.
 * Should call close method when finish using it.
 */
public class NeuropodTensorAllocator extends NativeClass {
    /**
     * Create a NeuropodTensor based on given ByteBuffer, dims array and tensorType. The created tensor
     * will have the input tensorType
     * <p>
     * Will not trigger a copy if the buffer is a direct bytebuffer in native order and the backend of
     * the allocator does not have alignment requirement. Otherwise it will trigger a copy.
     *
     * @param byteBuffer the buffer that contains tensor data
     * @param dims       the shape of the tensor
     * @param tensorType the tensor type
     * @return the created NeuropodTensor
     */
    NeuropodTensor create(ByteBuffer byteBuffer, long[] dims, TensorType tensorType) {
        return null;
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
}
