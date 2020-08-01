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

import java.util.HashMap;
import java.util.Map;

/**
 * This is an one to one mapping to cpp TensorType enum
 */
public enum TensorType {

    // Supported Types
    /**
     * The Float tensor.
     */
    FLOAT_TENSOR(0),
    /**
     * Double tensor data type.
     */
    DOUBLE_TENSOR(1),
    /**
     * Int 32 tensor data type.
     */
    INT32_TENSOR(2),
    /**
     * Int 64 tensor data type.
     */
    INT64_TENSOR(3),
    /**
     * String tensor data type.
     */
    STRING_TENSOR(4),

    // Unsupported Types, the types below are not supported by the java api
    /**
     * The Int 8 tensor.
     */
    INT8_TENSOR(5),
    /**
     * Int 16 tensor data type.
     */
    INT16_TENSOR(6),
    /**
     * Uint 8 tensor data type.
     */
    UINT8_TENSOR(7),
    /**
     * Uint 16 tensor data type.
     */
    UINT16_TENSOR(8),
    /**
     * Uint 32 tensor data type.
     */
    UINT32_TENSOR(9),
    /**
     * Uint 64 tensor data type.
     */
    UINT64_TENSOR(10);

    // These helper functions below are for supporting int to enum and enum to int conversion
    private int value;
    private static Map<Integer, TensorType> map = new HashMap<>();

    private TensorType(int value) {
        this.value = value;
    }

    static {
        for (TensorType tensorType : TensorType.values()) {
            map.put(tensorType.value, tensorType);
        }
    }

    protected static TensorType valueOf(int dataType) {
        return map.get(dataType);
    }

    protected int getValue() {
        return value;
    }
}
