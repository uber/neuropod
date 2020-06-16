package org.neuropod;

import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * This class is an one to one mapping to cpp NeuropodValue type.Need to manually call close
 * Method after using this class
 */
public class Neuropod extends NativeClass {

    static {
        LibraryLoader.load();
    }

    /**
     * Instantiates a new Neuropod.
     *
     * @param nativeHandle the native handle
     */
    public Neuropod(long nativeHandle) {
        super(nativeHandle);
    }

    /**
     * Instantiates a new Neuropod.
     *
     * @param filePath the file path
     */
    public Neuropod(String filePath) {
        super.setNativeHandle(nativeNew(filePath));
    }

    /**
     * Instantiates a new Neuropod.
     *
     * @param filePath the file path
     * @param options  the options
     */
    public Neuropod(String filePath, RuntimeOptions options) {
        super.setNativeHandle(nativeNew(filePath, options.getNativeHandle()));
    }

    /**
     * Infer neuropod value map.
     *
     * @param inputs the inputs
     * @return the neuropod value map
     * @throws Exception the exception
     */
    public NeuropodValueMap infer(Map<String, Object> inputs) throws Exception {
        checkInputSpec(inputs);
        NeuropodValueMap valueMap = new NeuropodValueMap(nativeCreateTensorsFromMemory(inputs, super.getNativeHandle()));
        NeuropodValueMap ret = infer(valueMap);
        valueMap.close();
        return ret;
    }

    /**
     * Infer neuropod value map.
     *
     * @param inputs the inputs
     * @return the neuropod value map
     */
    public synchronized NeuropodValueMap infer(NeuropodValueMap inputs) {
        return new NeuropodValueMap(nativeInfer(inputs.getNativeHandle(), super.getNativeHandle()));
    }

    // Check whether the input tensor names are the same with the model's requirement
    private void checkInputSpec(Map<String, Object> inputs) throws NeuropodJNIException {
        List<String> inputFeatureKeyList = nativeGetInputFeatureKeys(super.getNativeHandle());
        List<DataType> inputFeatureDataTypes = nativeGetInputFeatureDataTypes(super.getNativeHandle());
        Iterator<String> keyIt = inputFeatureKeyList.iterator();
        Iterator<DataType> typeIt = inputFeatureDataTypes.iterator();
        while (keyIt.hasNext() && typeIt.hasNext()) {
            String currentKey = keyIt.next();
            if (!inputs.containsKey(currentKey)) {
                throw new NeuropodJNIException(String.format("Neuropod Error: Tensor name(s) '{%s}' are not found in the input spec", currentKey));
            }
            DataType type = typeIt.next();
            if (type == DataType.STRING_TENSOR) {
                if (!(inputs.get(currentKey) instanceof String)) {
                    throw new NeuropodJNIException(String.format("Neuropod Error: Tensor name(s) '{%s}' should be a string", currentKey));
                }
            } else {
                if (!(inputs.get(currentKey) instanceof List)) {
                    throw new NeuropodJNIException(String.format("Neuropod Error: Tensor name(s) '{%s}' should have an array as its input data", currentKey));
                } else {
                    List<?> feature = (List<?>) inputs.get(currentKey);
                    if (feature.size() == 0) {
                        throw new NeuropodJNIException(String.format("Neuropod Error: Tensor name(s) '{%s}' should have a non emoty array as its input data", currentKey));
                    }
                    verifyType(feature.get(0), type, currentKey);
                }
            }
        }
    }

    // Check whether the input tensor has the same data type with the model's reuqirement
    private void verifyType(Object obj, DataType type, String tensorKey) {
        switch (type) {
            case FLOAT_TENSOR:
                if (!(obj instanceof Float)) {
                    throw new NeuropodJNIException(String.format("Neuropod Error: Tensor name(s) '{%s}' should have a float list as its input data", tensorKey));
                }
                break;
            case DOUBLE_TENSOR:
                if (!(obj instanceof Double)) {
                    throw new NeuropodJNIException(String.format("Neuropod Error: Tensor name(s) '{%s}' should have a double list as its input data", tensorKey));
                }
                break;
            case INT32_TENSOR:
                if (!(obj instanceof Integer)) {
                    throw new NeuropodJNIException(String.format("Neuropod Error: Tensor name(s) '{%s}' should have a int32 list as its input data", tensorKey));
                }
                break;
            case INT64_TENSOR:
                if (!(obj instanceof Long)) {
                    throw new NeuropodJNIException(String.format("Neuropod Error: Tensor name(s) '{%s}' should have a int64 list as its input data", tensorKey));
                }
                break;
            default:
                throw new NeuropodJNIException(String.format("Neuropod Error: Tensor name(s) '{%s}' has a unsupported data type", tensorKey));
        }

    }

    static private native List<String> nativeGetInputFeatureKeys(long handle);

    static private native List<DataType> nativeGetInputFeatureDataTypes(long handle);

    static private native long nativeNew(String filePath);

    static private native long nativeNew(String filePath, long optionHandle);

    static private native long nativeInfer(long inputHanlde, long modelHandle);

    static private native long nativeCreateTensorsFromMemory(Map<String, Object> data, long neuropodHandle);

    @Override
    protected native void nativeDelete(long handle);
}
