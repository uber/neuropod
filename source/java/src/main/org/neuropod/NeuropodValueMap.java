package org.neuropod;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This class is an one to one mapping to cpp NeuropodValueMap type,
 * an unordered_map[string, NeuropodValue]. Need to manually call close
 * Method after using this class
 */
public class NeuropodValueMap extends NativeClass {
    /**
     * Instantiates a new Neuropod value map from existing cpp handle.
     *
     * @param nativeHandle the native handle
     */
    public NeuropodValueMap(long nativeHandle) {
        super(nativeHandle);
    }

    /**
     * Instantiates a new empty Neuropod value map.
     */
    public NeuropodValueMap() {
        super(NeuropodValueMap.nativeNew());
    }

    /**
     * Equivalent to get method of java Map class, get the value from native unordered_map. This is a shallow copy,
     * the underlying native NeuropodValue is still owned by NeuropodValueMap. Once the NeuropodValueMap is closed,
     * the returned NeuropodValue becomes a wild reference.
     * TODO: Fix the messy ownership
     *
     * @param key the key
     * @return the value
     */
    public NeuropodValue getValue(String key) {
        return new NeuropodValue(nativeGetValue(key, super.getNativeHandle()));
    }


    /**
     * Equivalent to put method of java Map class, put the key value pair to native unordered_map. This is a shallow
     * copy, but the NeuropodValueMap will also have the ownership of the NeuropodValue.
     *
     * @param key           the key
     * @param neuropodValue the neuropod value
     */
    public void addEntry(String key, NeuropodValue neuropodValue) {
        nativeAddEntry(key, neuropodValue.getNativeHandle(), super.getNativeHandle());
    }

    /**
     * Equivalent to values method of java Map class, get value set from the native unordered_map. This is a shallow copy,
     * the underlying native NeuropodValue is still owned by NeuropodValueMap. Once the NeuropodValueMap is closed,
     * the returned NeuropodValue becomes a wild reference.
     *
     * @return the value list
     */
    public List<NeuropodValue> getValueList() {
        List<NeuropodValue> ret = new ArrayList<>();
        List<String> keys = getKeyList();
        for (String key : keys) {
            ret.add(getValue(key));
        }
        return ret;
    }

    /**
     * Equivalent to keySet method of java Map class, get the key set from the native unordered_map. This is a shallow copy,
     *
     * @return the key list
     */
    public List<String> getKeyList() {
        return nativeGetKeyList(super.getNativeHandle());
    }

    /**
     * Copy the data in native unordered_map to a java map. The data in NeuropodValue will be flattened.
     *
     * @return the map
     */
    public Map<String, Object> toJavaMap() {
        List<String> keys = getKeyList();
        Map<String, Object> ret = new HashMap<>();
        for (String key : keys) {
            ret.put(key, getValue(key).toList());
        }
        return ret;
    }

    @Override
    protected native void nativeDelete(long handle);

    static private native long nativeNew() throws NeuropodJNIException;

    static private native void nativeAddEntry(String key, long valueNativeHandle, long mapNativeHandle) throws NeuropodJNIException;

    static private native long nativeGetValue(String key, long handle) throws NeuropodJNIException;

    static private native List<String> nativeGetKeyList(long handle) throws NeuropodJNIException;

}
