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
import org.junit.Assert;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.*;

import static org.junit.Assert.*;

// Ignoring base class becuase derived will run test.
public class NeuropodAdditionTest {
    protected Neuropod model;
    // Model path should be set by derived class before setUp.
    protected String model_path;

    // Platform should be set by derived class to (tensorflow, torchscript).
    protected String platform;

    // When asserting float and double, juint4 requires an additional delta value. Float and double are not precise.
    // If the absolute difference between the expected value and the actual value is smaller than this delta value,
    // junit4 will think they are the same value even they do not equal to each other.
    private static final double EPSILON = 1E-6;

    protected void prepareEnvironment() throws Exception {
        LibraryLoader.load();
        // Set the test mode to true to use override library path
        LibraryLoader.setTestMode(true);
        RuntimeOptions opts = new RuntimeOptions();
        // TODO(weijiad): For now OPE is required to use the Java bindings
        opts.useOpe = true;
        model = new Neuropod(model_path, opts);
    }

    @Test
    public void testClosed() {
        model.close();
        try {
            model.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
            fail("Exception is not expected");
        }

        try {
            model.getName();
            fail("Exception is expected because model is closed already");
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    @Test
    public void getName() {
        String name = model.getName();
        assertEquals("addition_model", name);
    }

    @Test
    public void getPlatform() {
        String platform = model.getPlatform();
        assertEquals(this.platform, platform);
    }

    @Test
    public void getInputs() {
        Set<TensorSpec> inputs = new HashSet<>(model.getInputs());
        Set<TensorSpec> expected = new HashSet<>(Arrays.asList(
                new TensorSpec("x", TensorType.FLOAT_TENSOR,
                        Arrays.asList(new Dimension(-1), new Dimension(-1))),
                new TensorSpec("y", TensorType.FLOAT_TENSOR,
                        Arrays.asList(new Dimension(-1), new Dimension(-1)))));
        assertEquals(inputs, expected);
    }

    @Test
    public void getOutputs() {
        Set<TensorSpec> outputs = new HashSet<>(model.getOutputs());
        Set<TensorSpec> expected = new HashSet<>(Arrays.asList(
                new TensorSpec("out", TensorType.FLOAT_TENSOR,
                        Arrays.asList(new Dimension(-1), new Dimension(-1)))));
        assertEquals(outputs, expected);
    }

    @Test
    public void loadModel() {
        RuntimeOptions ope = new RuntimeOptions();
        // TODO(weijiad): For now OPE is required to use the Java bindings
        ope.useOpe = true;
        ope.loadModelAtConstruction = false;
        try (Neuropod model = new Neuropod(model_path, ope)) {
            try {
                model.loadModel();
                NeuropodTensorAllocator allocator = model.getTensorAllocator();
                allocator.close();
            } catch (Exception e) {
                System.out.println(e.getMessage());
                fail();
            }
        }
    }

    @Test
    public void InputDoubleTensor() {
       NeuropodTensorAllocator allocator = model.getTensorAllocator();

       TensorType type = TensorType.DOUBLE_TENSOR;
       ByteBuffer buffer = ByteBuffer.allocateDirect(type.getBytesPerElement() * 2).order(ByteOrder.nativeOrder());

       DoubleBuffer typedBuffer = buffer.asDoubleBuffer();
       typedBuffer.put(1.0f);
       typedBuffer.put(2.0f);
       NeuropodTensor tensor = allocator.tensorFromMemory(buffer, new long[]{1L, 2L}, type);

       // TBD: will be implemented next.
       // assertNotNull(tensor.toDoubleBuffer());
       assertNull(tensor.toDoubleBuffer());

       assertArrayEquals(new long[]{1L, 2L}, tensor.getDims());
       assertEquals(2, tensor.getNumberOfElements());
       assertEquals(TensorType.DOUBLE_TENSOR, tensor.getTensorType());

       tensor.close();
       allocator.close();
    }

    @Test
    public void InputIntTensor() {
       NeuropodTensorAllocator allocator = model.getTensorAllocator();

       TensorType type = TensorType.INT32_TENSOR;
       ByteBuffer buffer = ByteBuffer.allocateDirect(type.getBytesPerElement() * 2).order(ByteOrder.nativeOrder());

       IntBuffer typedBuffer = buffer.asIntBuffer();
       typedBuffer.put(1);
       typedBuffer.put(2);
       NeuropodTensor tensor = allocator.tensorFromMemory(buffer, new long[]{1L, 2L}, type);

       // TBD: will be implemented next.
       // assertNotNull(tensor.toIntBuffer());
       assertNull(tensor.toIntBuffer());

       assertArrayEquals(new long[]{1L, 2L}, tensor.getDims());
       assertEquals(2, tensor.getNumberOfElements());
       assertEquals(TensorType.INT32_TENSOR, tensor.getTensorType());

       tensor.close();
       allocator.close();
    }

    @Test
    public void InputLongTensor() {
       NeuropodTensorAllocator allocator = model.getTensorAllocator();

       TensorType type = TensorType.INT64_TENSOR;
       ByteBuffer buffer = ByteBuffer.allocateDirect(type.getBytesPerElement() * 2).order(ByteOrder.nativeOrder());

       LongBuffer typedBuffer = buffer.asLongBuffer();
       typedBuffer.put(1);
       typedBuffer.put(2);
       NeuropodTensor tensor = allocator.tensorFromMemory(buffer, new long[]{1L, 2L}, type);
       assertNotNull(tensor.toLongBuffer());

       assertArrayEquals(new long[]{1L, 2L}, tensor.getDims());
       assertEquals(2, tensor.getNumberOfElements());
       assertEquals(TensorType.INT64_TENSOR, tensor.getTensorType());

       tensor.close();
       allocator.close();
    }

    @Test
    public void infer() {
        NeuropodTensorAllocator allocator = model.getTensorAllocator();
        Map<String, NeuropodTensor> inputs = new HashMap<>();
        TensorType type = TensorType.FLOAT_TENSOR;

        ByteBuffer bufferX = ByteBuffer.allocateDirect(type.getBytesPerElement() * 2).order(ByteOrder.nativeOrder());
        FloatBuffer floatBufferX = bufferX.asFloatBuffer();
        floatBufferX.put(1.0f);
        floatBufferX.put(2.0f);
        NeuropodTensor tensorX = allocator.tensorFromMemory(bufferX, new long[]{1L, 2L}, type);
        inputs.put("x", tensorX);

        ByteBuffer bufferY = ByteBuffer.allocateDirect(type.getBytesPerElement() * 2).order(ByteOrder.nativeOrder());
        FloatBuffer floatBufferY = bufferY.asFloatBuffer();
        floatBufferY.put(3.0f);
        floatBufferY.put(4.0f);
        NeuropodTensor tensorY = allocator.tensorFromMemory(bufferY, new long[]{1L, 2L}, type);
        inputs.put("y", tensorY);

        Map<String, NeuropodTensor> res = model.infer(inputs);
        assertEquals(1, res.size());

        assertTrue(res.containsKey("out"));
        NeuropodTensor out = res.get("out");
        assertNotNull(out);
        FloatBuffer outBuffer = out.toFloatBuffer();

        assertArrayEquals(new long[]{1L, 2L}, out.getDims());
        assertEquals(2, out.getNumberOfElements());
        assertEquals(TensorType.FLOAT_TENSOR, out.getTensorType());

        assertEquals(4.0f , outBuffer.get(0), EPSILON);
        assertEquals(6.0f , outBuffer.get(1), EPSILON);

        out.close();

        // Inference with requested outputs.
        List<String> requestedOutputs = new ArrayList<String>();
        requestedOutputs.add("out");
        Map<String, NeuropodTensor> res2 = model.infer(inputs, requestedOutputs);
        assertEquals(1, res2.size());

        tensorX.close();
        tensorY.close();
        allocator.close();
    }

    @Test
    public void inferWithUnexpectedRequestedOutput() {
        // Note that TF performs validation of requested output at the very begining.
        // Hence it doesn't need valid Input in this test because it supposed to fail before it tests Input.
        // But Torchscript is trying to use input and only then detects wrong requested output.
        // This is why we build valid Input here even we don't care about input/output value really.
        NeuropodTensorAllocator allocator = model.getTensorAllocator();
        TensorType type = TensorType.FLOAT_TENSOR;
        ByteBuffer buffer = ByteBuffer.allocateDirect(type.getBytesPerElement() * 2).order(ByteOrder.nativeOrder());
        FloatBuffer floatBuffer = buffer.asFloatBuffer();
        floatBuffer.put(1.0f);
        NeuropodTensor tensor = allocator.tensorFromMemory(buffer, new long[]{1L, 1L}, type);

        Map<String, NeuropodTensor> inputs = new HashMap<>();
        inputs.put("x", tensor);
        inputs.put("y", tensor);

        try
        {
           // Inference with requested outputs.
           List<String> requestedOutputs = new ArrayList<String>();
           requestedOutputs.add("out");
           // Add unexpected "requested output" that should cause a failure.
           requestedOutputs.add("out_wrong");
           Map<String, NeuropodTensor> res = model.infer(inputs, requestedOutputs);
           Assert.fail("Expected exception on wrong requested output");
        }
        catch (Exception expected)
        {
           // Note that TF and Torchscript returns different exception message.
           // TF: Node out_wrong not found in node_name_mapping
           // Torchscript: Tried to request a tensor that does not exist: out_wrong
           // Test that message contains out_wrong name.
           assertTrue(expected.getMessage(), expected.getMessage().contains("out_wrong"));
        }

        tensor.close();
        allocator.close();
    }

    @After
    public void tearDown() throws Exception {
        model.close();
    }
}
