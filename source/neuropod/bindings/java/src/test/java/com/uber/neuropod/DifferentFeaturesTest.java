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

import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

public class DifferentFeaturesTest {
    @Before
    public void setUp() throws Exception {
        LibraryLoader.load();
        // Set the test mode to true to use override library path
        LibraryLoader.setTestMode(true);
    }

    @Test
    public void getOutputsTestSymbol() {
        // Load model that we know has named dimensions.
        final String modelPath = "neuropod/tests/test_data/torchscript_addition_model_single_output/";
        RuntimeOptions opts = new RuntimeOptions();
        opts.useOpe = true;
        try (Neuropod torchModel = new Neuropod(modelPath, opts)) {
            Set<TensorSpec> outputs = new HashSet<>(torchModel.getOutputs());
            Set<TensorSpec> expected = new HashSet<>(Arrays.asList(
                    new TensorSpec("out", TensorType.FLOAT_TENSOR,
                            Arrays.asList(new Dimension("batch_size"), new Dimension(-1)))));
            assertEquals(outputs, expected);
        }
    }
}
