package com.uber.neuropod;

import java.util.List;
import java.util.Map;

/**
 * This class holds the information of a Neuropod model. It has its underlying C++ Neuropod
 * object, should call close() function to free the C++ side object when finish using the Neuopod object
 */
public class Neuropod extends NativeClass {

    /**
     * Load a model from the file path and use default options
     *
     * @param neuropodPath the neuropod model path
     */
    public Neuropod(String neuropodPath) {}

    /**
     * Load a model from the file path and use provided runtime options
     *
     * @param neuropodPath the neuropod model path
     * @param options      the runtime options
     */
    public Neuropod(String neuropodPath, RuntimeOptions options) { }

    /**
     * Gets the model name.
     *
     * @return the name
     */
    public String getName() {
        return null;
    }

    /**
     * Gets the platform name.
     *
     * @return the platform
     */
    public String getPlatform() {
        return null;
    }


    /**
     * Perform the inference calculation on the given input data
     *
     * @param inputs the inputs
     * @return the inference result
     */
    public Map<String, NeuropodTensor> infer(Map<String,NeuropodTensor> inputs) {
        return null;
    }

    /**
     * Perform the inference calculation on the given input data, only get output tensor
     * with the keys specified by requestOutputs.
     *
     * @param inputs         the inputs
     * @param requestedOutputs only tensor with these keys will be output
     * @return the inference result
     */
    public Map<String, NeuropodTensor> infer(Map<String, NeuropodTensor> inputs, List<String> requestedOutputs) {
        return null;
    }

    /**
     * Gets input tensor specs
     *
     * @return the input specs
     */
    public List<TensorSpec> getInputs() {
        return null;
    }

    /**
     * Gets output tensor specs
     *
     * @return the output specs
     */
    public List<TensorSpec> getOutputs() {
        return null;
    }

    /**
     * Load model. Used when setLoadModelAtConstruction is set to false in RuntimeOptions
     */
    public void loadModel() {}

    /**
     * Gets a tensor allocator based on the backend of the model.
     *
     * @return the tensor allocator
     */
    public NeuropodTensorAllocator getTensorAllocator() {return null;}

    /**
     * Gets a generic tensor allocator.
     *
     * @return the generic tensor allocator
     */
    public static NeuropodTensorAllocator getGenericTensorAllocator() {return null;}
}
