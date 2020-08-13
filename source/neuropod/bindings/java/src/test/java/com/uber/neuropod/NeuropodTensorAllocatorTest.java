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

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.nio.*;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class NeuropodTensorAllocatorTest {
    private NeuropodTensorAllocator allocator;

    @Before
    public void setUp() throws Exception {
        allocator = Neuropod.getGenericTensorAllocator();
    }

    @Test
    public void tensorFromMemory() {
        TensorType type = TensorType.INT64_TENSOR;
        ByteBuffer buffer = ByteBuffer.allocateDirect(4 * type.getBytesPerElement()).order(ByteOrder.nativeOrder());
        LongBuffer longBuffer = buffer.asLongBuffer();
        for (int i = 0; i < 4; i++) {
            longBuffer.put(i);
        }

        long[] shape = new long[]{2, 2};
        try (NeuropodTensor tensor = allocator.tensorFromMemory(buffer, shape, type)) {
            assertArrayEquals(shape, tensor.getDims());
            assertEquals(4, tensor.getNumberOfElements());
            assertEquals(TensorType.INT64_TENSOR, tensor.getTensorType());

            LongBuffer outputBuffer = tensor.toLongBuffer();
            for (int i = 0; i < 4; i++) {
                assertEquals(longBuffer.get(i), outputBuffer.get(i));
            }
        }
    }

    @After
    public void tearDown() throws Exception {
        allocator.close();
    }
}
