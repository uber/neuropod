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

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.*;

public class NeuropodTest {
    private Neuropod model;
    private static final String TF_MODEL_PATH = "neuropod/tests/test_data/tf_addition_model/";
    private static final String TORCHSCRIPT_MODEL_PATH = "neuropod/tests/test_data/torchscript_addition_model_single_output/";

    @Before
    public void setUp() throws Exception {
        LibraryLoader.load();
        // Set the test mode to true to use override library path
        LibraryLoader.setTestMode(true);
        RuntimeOptions opts = new RuntimeOptions();
        // TODO(weijiad): For now OPE is required to use the Java bindings
        opts.useOpe = true;
        model = new Neuropod(TF_MODEL_PATH, opts);
    }

    @Test
    public void testClosed() {
        model.close();
        try {
            model.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
            fail();
        }
        try {
            model.getName();
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return;
        }
        fail();
    }

    @Test
    public void getName() {
        String name = model.getName();
        assertEquals("addition_model", name);
    }

    @Test
    public void getPlatform() {
        String platform = model.getPlatform();
        assertEquals("tensorflow", platform);
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
    public void getOutputsTestSymbol() {
        RuntimeOptions opts = new RuntimeOptions();
        // TODO(weijiad): For now OPE is required to use the Java bindings
        opts.useOpe = true;
        try (Neuropod torchModel = new Neuropod(TORCHSCRIPT_MODEL_PATH, opts)) {
            Set<TensorSpec> outputs = new HashSet<>(torchModel.getOutputs());
            Set<TensorSpec> expected = new HashSet<>(Arrays.asList(
                    new TensorSpec("out", TensorType.FLOAT_TENSOR,
                            Arrays.asList(new Dimension("batch_size"), new Dimension(-1)))));
        }
    }

    @Test
    public void loadModel() {
        RuntimeOptions ope = new RuntimeOptions();
        // TODO(weijiad): For now OPE is required to use the Java bindings
        ope.useOpe = true;
        ope.loadModelAtConstruction = false;
        try (Neuropod model = new Neuropod(TF_MODEL_PATH, ope)) {
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

    @After
    public void tearDown() throws Exception {
        model.close();
    }
}
