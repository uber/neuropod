/* Copyright (c) 2020 The Neuropod Authors

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
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.nio.*;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

public class NeuropodTensorAllocatorTest {
    private NeuropodTensorAllocator allocator;
    private static final double EPSILON = 1E-6;

    @Before
    public void setUp() throws Exception {
        allocator = Neuropod.getGenericTensorAllocator();
    }

    @Test
    public void getTensorValueAsIntWithOutOfBounds() {
        IntBuffer intBuffer = IntBuffer.allocate(4);
        for (int i = 0; i < 4; i++) {
            intBuffer.put(i);
        }
        intBuffer.rewind();
        long[] shape = new long[] {2, 2};

        try (NeuropodTensor tensor = allocator.copyFrom(intBuffer, shape)) {
           int value = tensor.getInt(2, 10);
           Assert.fail("Expected exception on incorect indexes");
        } catch (Exception expected) {
           assertTrue(expected.getMessage(), expected.getMessage().contains("has out of bounds value"));
        }

        try (NeuropodTensor tensor = allocator.copyFrom(intBuffer, shape)) {
           int value = tensor.getInt(1, 2, 3, 2, 10);
           Assert.fail("Expected exception on incorect indexes");
        } catch (Exception expected) {
           assertTrue(expected.getMessage(), expected.getMessage().contains("does not match dimension size"));
        }
    }

    @Test
    public void copyFromInt() {
        IntBuffer intBuffer = IntBuffer.allocate(4);
        for (int i = 0; i < 4; i++) {
            intBuffer.put(i);
        }
        intBuffer.rewind();
        long[] shape = new long[] {2, 2};
        try (NeuropodTensor tensor = allocator.copyFrom(intBuffer, shape)) {
            assertArrayEquals(tensor.getDims(), shape);
            assertEquals(4, tensor.getNumberOfElements());
            assertEquals(TensorType.INT32_TENSOR, tensor.getTensorType());
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    assertEquals(intBuffer.get(i * 2 + j), tensor.getInt(i, j));
                }
            }
            IntBuffer outputBuffer = tensor.toIntBuffer();
            for (int i = 0; i < 4; i++) {
                assertEquals(intBuffer.get(i), outputBuffer.get(i));
            }
        }
    }

    @Test
    public void copyFromLong() {
        LongBuffer longBuffer = LongBuffer.allocate(4);
        for (int i = 0; i < 4; i++) {
            longBuffer.put(i);
        }
        longBuffer.rewind();
        long[] shape = new long[] {2, 2};
        try (NeuropodTensor tensor = allocator.copyFrom(longBuffer, shape)) {
            assertArrayEquals(tensor.getDims(), shape);
            assertEquals(4, tensor.getNumberOfElements());
            assertEquals(TensorType.INT64_TENSOR, tensor.getTensorType());
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    assertEquals(longBuffer.get(i * 2 + j), tensor.getLong(i, j));
                }
            }
            LongBuffer outputBuffer = tensor.toLongBuffer();
            for (int i = 0; i < 4; i++) {
                assertEquals(longBuffer.get(i), outputBuffer.get(i));
            }
        }
    }

    @Test
    public void copyFromFloat() {
        FloatBuffer floatBuffer = FloatBuffer.allocate(4);
        for (int i = 0; i < 4; i++) {
            floatBuffer.put(i);
        }
        floatBuffer.rewind();
        long[] shape = new long[] {2, 2};
        try (NeuropodTensor tensor = allocator.copyFrom(floatBuffer, shape)) {
            assertArrayEquals(tensor.getDims(), shape);
            assertEquals(4, tensor.getNumberOfElements());
            assertEquals(TensorType.FLOAT_TENSOR, tensor.getTensorType());
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    assertEquals(floatBuffer.get(i * 2 + j), tensor.getFloat(i, j), EPSILON);
                }
            }
            FloatBuffer outputBuffer = tensor.toFloatBuffer();
            for (int i = 0; i < 4; i++) {
                assertEquals(floatBuffer.get(i), outputBuffer.get(i), EPSILON);
            }
        }
    }

    @Test
    public void copyFromDouble() {
        DoubleBuffer doubleBuffer = DoubleBuffer.allocate(4);
        for (int i = 0; i < 4; i++) {
            doubleBuffer.put(i);
        }
        doubleBuffer.rewind();
        long[] shape = new long[] {2, 2};
        try (NeuropodTensor tensor = allocator.copyFrom(doubleBuffer, shape)) {
            assertArrayEquals(tensor.getDims(), shape);
            assertEquals(4, tensor.getNumberOfElements());
            assertEquals(TensorType.DOUBLE_TENSOR, tensor.getTensorType());
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    assertEquals(doubleBuffer.get(i * 2 + j), tensor.getDouble(i, j), EPSILON);
                }
            }
            DoubleBuffer outputBuffer = tensor.toDoubleBuffer();
            for (int i = 0; i < 4; i++) {
                assertEquals(doubleBuffer.get(i), outputBuffer.get(i), EPSILON);
            }

        }
    }

    @Test
    public void copyFromString() {
        List<String> input = Arrays.asList("abc", "def", "ghi", "jkl");
        long[] shape = new long[] {2, 2};
        try (NeuropodTensor tensor = allocator.copyFrom(input, shape)) {
            assertArrayEquals(tensor.getDims(), shape);
            assertEquals(4, tensor.getNumberOfElements());
            assertEquals(TensorType.STRING_TENSOR, tensor.getTensorType());
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    assertEquals(input.get(i * 2 + j), tensor.getString(i, j));
                }
            }
            List<String> output = tensor.toStringList();
            for (int i = 0; i < 4; i++) {
                assertEquals(input.get(i), output.get(i));
            }
        }
    }

    @Test
    public void tensorFromMemoryFloat() {
        TensorType type = TensorType.FLOAT_TENSOR;
        ByteBuffer buffer = ByteBuffer.allocateDirect(4 * type.getBytesPerElement())
                .order(ByteOrder.nativeOrder());
        FloatBuffer typedBuffer = buffer.asFloatBuffer();
        for (int i = 0; i < 4; i++) {
            typedBuffer.put(i);
        }

        long[] shape = new long[] {2, 2};
        try (NeuropodTensor tensor = allocator.tensorFromMemory(buffer, shape, type)) {
            assertArrayEquals(shape, tensor.getDims());
            assertEquals(4, tensor.getNumberOfElements());
            assertEquals(TensorType.FLOAT_TENSOR, tensor.getTensorType());

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    assertEquals(tensor.getFloat(i, j), typedBuffer.get(i * 2 + j), EPSILON);
                }
            }

            FloatBuffer outputBuffer = tensor.toFloatBuffer();
            for (int i = 0; i < 4; i++) {
                assertEquals(typedBuffer.get(i), outputBuffer.get(i), EPSILON);
            }
        }
    }

    @Test
    public void tensorFromMemoryDouble() {
        TensorType type = TensorType.DOUBLE_TENSOR;
        ByteBuffer buffer = ByteBuffer.allocateDirect(4 * type.getBytesPerElement())
                .order(ByteOrder.nativeOrder());
        DoubleBuffer typedBuffer = buffer.asDoubleBuffer();
        for (int i = 0; i < 4; i++) {
            typedBuffer.put(i);
        }

        long[] shape = new long[] {2, 2};
        try (NeuropodTensor tensor = allocator.tensorFromMemory(buffer, shape, type)) {
            assertArrayEquals(shape, tensor.getDims());
            assertEquals(4, tensor.getNumberOfElements());
            assertEquals(TensorType.DOUBLE_TENSOR, tensor.getTensorType());

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    assertEquals(tensor.getDouble(i, j), typedBuffer.get(i * 2 + j), EPSILON);
                }
            }

            DoubleBuffer outputBuffer = tensor.toDoubleBuffer();
            for (int i = 0; i < 4; i++) {
                assertEquals(typedBuffer.get(i), outputBuffer.get(i), EPSILON);
            }
        }
    }

    @Test
    public void tensorFromMemoryInt() {
        TensorType type = TensorType.INT32_TENSOR;
        ByteBuffer buffer = ByteBuffer.allocateDirect(4 * type.getBytesPerElement())
                .order(ByteOrder.nativeOrder());
        IntBuffer typedBuffer = buffer.asIntBuffer();
        for (int i = 0; i < 4; i++) {
            typedBuffer.put(i);
        }

        long[] shape = new long[] {2, 2};
        try (NeuropodTensor tensor = allocator.tensorFromMemory(buffer, shape, type)) {
            assertArrayEquals(shape, tensor.getDims());
            assertEquals(4, tensor.getNumberOfElements());
            assertEquals(TensorType.INT32_TENSOR, tensor.getTensorType());

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    assertEquals(tensor.getInt(i, j), typedBuffer.get(i * 2 + j), EPSILON);
                }
            }

            IntBuffer outputBuffer = tensor.toIntBuffer();
            for (int i = 0; i < 4; i++) {
                assertEquals(typedBuffer.get(i), outputBuffer.get(i));
            }
        }
    }

    @Test
    public void tensorFromMemoryLong() {
        TensorType type = TensorType.INT64_TENSOR;
        ByteBuffer buffer = ByteBuffer.allocateDirect(4 * type.getBytesPerElement())
                .order(ByteOrder.nativeOrder());
        LongBuffer typedBuffer = buffer.asLongBuffer();
        for (int i = 0; i < 4; i++) {
            typedBuffer.put(i);
        }

        long[] shape = new long[] {2, 2};
        try (NeuropodTensor tensor = allocator.tensorFromMemory(buffer, shape, type)) {
            assertArrayEquals(shape, tensor.getDims());
            assertEquals(4, tensor.getNumberOfElements());
            assertEquals(TensorType.INT64_TENSOR, tensor.getTensorType());

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    assertEquals(tensor.getLong(i, j), typedBuffer.get(i * 2 + j), EPSILON);
                }
            }

            LongBuffer outputBuffer = tensor.toLongBuffer();
            for (int i = 0; i < 4; i++) {
                assertEquals(typedBuffer.get(i), outputBuffer.get(i));
            }
        }
    }

    @Test
    public void tensorFromMemoryString() {
       TensorType type = TensorType.STRING_TENSOR;
       // Note that for String Tensor we can't use low level allocator's method tensorFromMemory
       // that accepts direct ByteBuffer as input. It throws exception "unsupported tensor type"
       // and intentionally prevents using it for String Tensors.

       ByteBuffer buffer = ByteBuffer.allocateDirect("String Tensor Value".length())
                .order(ByteOrder.nativeOrder());
       // Note that we theoretically can write String data into buffer but this is
       // not required for this test, so keep buffer data as is.
        long[] shape = new long[] {2, 2};
        try (NeuropodTensor tensor = allocator.tensorFromMemory(buffer, shape, type)) {
            // it should not be able to initialize correctly and
            // hence never executes body of try-with block.
            fail("Expected exception 'unsupported tensor type'");
        } catch (Exception expected) {
           // com.uber.neuropod.NeuropodJNIException: unsupported tensor type: STRING_TENSOR
           assertTrue(expected.getMessage(), expected.getMessage().contains("unsupported tensor type"));
        }

       allocator.close();
    }

    @After
    public void tearDown() throws Exception {
        allocator.close();
    }
}
