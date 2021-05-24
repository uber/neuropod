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

import java.util.List;
import java.util.Map;

/**
 * This class holds the information of a Neuropod model. It has its underlying C++ Neuropod
 * object, should call close() function to free the C++ side object when finish using the Neuropod object
 * This class is not thread-safe.
 */
public class Neuropod extends NativeClass {

    protected Neuropod(long nativeHandle) {
        super(nativeHandle);
    }

    /**
     * Load a model from the file path and use default options
     *
     * @param neuropodPath the neuropod model path
     */
    public Neuropod(String neuropodPath) throws NeuropodJNIException, IllegalAccessException {
        super.setNativeHandle(nativeNew(neuropodPath, 0));
    }

    /**
     * Load a model from the file path and use provided runtime options
     *
     * @param neuropodPath the neuropod model path
     * @param options      the runtime options
     */
    public Neuropod(String neuropodPath, RuntimeOptions options) throws NeuropodJNIException {
        RuntimeOptions.RuntimeOptionsNative nativeOptions = options.toNative();
        super.setNativeHandle(nativeNew(neuropodPath, nativeOptions.getNativeHandle()));
        nativeOptions.close();
    }

    /**
     * Gets the model name.
     *
     * @return the name
     */
    public String getName() {
        return nativeGetName(super.getNativeHandle());
    }

    /**
     * Gets the platform name.
     *
     * @return the platform
     */
    public String getPlatform() {
        return nativeGetPlatform(super.getNativeHandle());
    }

    /**
     * Perform the inference calculation on the given input data
     *
     * @param inputs the inputs
     * @return the inference result
     */
    public Map<String, NeuropodTensor> infer(Map<String, NeuropodTensor> inputs) {
        return infer(inputs, null);
    }

    /**
     * Perform the inference calculation on the given input data, only get output tensor
     * with the keys specified by requestOutputs.
     *
     * @param inputs           the inputs
     * @param requestedOutputs only tensor with these keys will be output
     * @return the inference result
     */
    public Map<String, NeuropodTensor> infer(Map<String, NeuropodTensor> inputs, List<String> requestedOutputs) {
        return nativeInfer(inputs.entrySet().toArray(), requestedOutputs, super.getNativeHandle());
    }

    /**
     * Gets input tensor specs
     *
     * @return the input specs
     */
    public List<TensorSpec> getInputs() {
        return nativeGetInputs(super.getNativeHandle());
    }

    /**
     * Gets output tensor specs
     *
     * @return the output specs
     */
    public List<TensorSpec> getOutputs() {
        return nativeGetOutputs(super.getNativeHandle());
    }

    /**
     * Load model. Used when setLoadModelAtConstruction is set to false in RuntimeOptions
     */
    public void loadModel() {
        nativeLoadModel(super.getNativeHandle());
    }

    /**
     * Gets a tensor allocator based on the backend of the model.
     *
     * @return the tensor allocator
     */
    public NeuropodTensorAllocator getTensorAllocator() {
        return new NeuropodTensorAllocator(nativeGetAllocator(super.getNativeHandle()));
    }

    /**
     * Gets a generic tensor allocator.
     *
     * @return the generic tensor allocator
     */
    public static NeuropodTensorAllocator getGenericTensorAllocator() {
        return new NeuropodTensorAllocator(nativeGetGenericAllocator());
    }

    private static native long nativeNew(String filePath, long optionHandle);

    private static native void nativeLoadModel(long modelHandle);

    private static native String nativeGetName(long modelHandle);

    private static native String nativeGetPlatform(long modelHandle);

    private static native List<TensorSpec> nativeGetInputs(long modelHandle);

    private static native List<TensorSpec> nativeGetOutputs(long modelHandle);

    private static native long nativeGetAllocator(long modelHandle);

    private static native long nativeGetGenericAllocator();

    private static native Map<String, NeuropodTensor> nativeInfer(Object[] inputs, List<String> requestedOutputs,
                                                                  long modelHandle);
    @Override
    protected native void nativeDelete(long handle);


    public class Prepared extends NativeClass {
        private Map<String, NeuropodTensor> inputsArray = null;
        Prepared(Map<String, NeuropodTensor> inputs) {
            super(nativePrepare(inputs.entrySet().toArray()));
            this.inputsArray = inputs;
        }

        /**
         * Call callable by given handle. before this calls, Input Tensor buffers should be updated with new values.
         * After the call output data will be availbale through Output Tensors that were provided to callable
         * during "prepared" call.
         * @param handle to callable
         */
        public Map<String, NeuropodTensor> infer() {
            return Neuropod.nativeInferPrepared(super.getNativeHandle(), Neuropod.this.getNativeHandle());
        }

        @Override
        protected void nativeDelete(long handle) {
            if (this.inputsArray != null)
                this.inputsArray.values().forEach(NeuropodTensor::closeQuietly);
            nativeDeletePrepared(handle);
        }
    }

    // Returns handle for inference context with given inputs for model. Note that inputs already have Tensors
    // that were allocated with model's allocator and this is why this call doesn't need model's handle at all.
    // Caller can prepare it once, then update input buffers, call infer and get output.
    // This can be repeated again for the same set of inputs, with the same size and shape.
    private static native long nativePrepare(Object[] inputs);
    private static native void nativeDeletePrepared(long handle);

    // Perform inference using prepared  that already contains inputs.
    private static native Map<String, NeuropodTensor> nativeInferPrepared(long preparedHandle, long modelHandle);
}
