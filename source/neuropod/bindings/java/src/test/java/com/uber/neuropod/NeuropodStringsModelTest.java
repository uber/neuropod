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

import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

public class NeuropodStringsModelTest {
  protected Neuropod model;
  // Model path should be set by derived class before setUp.
  protected String model_path;

  // Platform should be set by derived class to (tensorflow, torchscript).
  protected String platform;

  protected void prepareEnvironment() throws Exception {
    LibraryLoader.load();

    RuntimeOptions opts = new RuntimeOptions();
    // TODO(haoyang): For now OPE is required to use the Java bindings
    opts.useOpe = true;
    model = new Neuropod(model_path, opts);
  }

  @Test
  public void infer() {
    NeuropodTensorAllocator allocator = model.getTensorAllocator();
    Map<String, NeuropodTensor> inputs = new HashMap<>();
    TensorType type = TensorType.STRING_TENSOR;

    List<String> bufferX = new ArrayList<String>();
    bufferX.add("applé🅫");
    bufferX.add("banana");
    NeuropodTensor tensorX = allocator.copyFrom(bufferX, new long[]{2L});
    inputs.put("x", tensorX);

    List<String> bufferY = new ArrayList<String>();
    bufferY.add("sauce");
    bufferY.add("pudding");
    NeuropodTensor tensorY = allocator.copyFrom(bufferY, new long[]{2L});
    inputs.put("y", tensorY);

    Map<String, NeuropodTensor> res = model.infer(inputs);
    assertEquals(1, res.size());

    assertTrue(res.containsKey("out"));
    NeuropodTensor out = res.get("out");
    assertNotNull(out);
    List<String> outStrings = out.toStringList();

    assertArrayEquals(new long[]{2L}, out.getDims());
    assertEquals(2, out.getNumberOfElements());
    assertEquals(TensorType.STRING_TENSOR, out.getTensorType());

    assertEquals("applé🅫 sauce" , outStrings.get(0));
    assertEquals("banana pudding" , outStrings.get(1));

    try {
      // Test that it detects type-mismatch if we try to take Float Output Tensor as Double.
      DoubleBuffer doubleBuffer = out.toDoubleBuffer();
      Assert.fail("Expected exception on wrong type");
    } catch (Exception expected) {
      assertTrue(expected.getMessage(), expected.getMessage().contains("tensorType mismatch"));
    }

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

}
