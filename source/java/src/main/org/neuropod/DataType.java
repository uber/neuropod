package org.neuropod;

/**
 * This is an one to one mapping to cpp TensorType enum
 */
public enum DataType {
    /**
     * The Float tensor.
     */
// Supported Types
    FLOAT_TENSOR,
    /**
     * Double tensor data type.
     */
    DOUBLE_TENSOR,
    /**
     * String tensor data type.
     */
    STRING_TENSOR,
    /**
     * Int 32 tensor data type.
     */
    INT32_TENSOR,
    /**
     * Int 64 tensor data type.
     */
    INT64_TENSOR,
    /**
     * The Int 8 tensor.
     */
// UnSupported Types, the types below are not supported by the java api
    INT8_TENSOR,
    /**
     * Int 16 tensor data type.
     */
    INT16_TENSOR,
    /**
     * Uint 8 tensor data type.
     */
    UINT8_TENSOR,
    /**
     * Uint 16 tensor data type.
     */
    UINT16_TENSOR,
    /**
     * Uint 32 tensor data type.
     */
    UINT32_TENSOR,
    /**
     * Uint 64 tensor data type.
     */
    UINT64_TENSOR
}
