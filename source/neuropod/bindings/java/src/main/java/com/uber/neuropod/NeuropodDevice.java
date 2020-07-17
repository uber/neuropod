package com.uber.neuropod;

import java.util.HashMap;
import java.util.Map;

/**
 * Which device the model uses. Use const int instead of enum to
 * support number greater than 7, or namely GPU8 or more GPUs
 */
public class NeuropodDevice {
    public static final int CPU = -1;
    public static final int GPU0 = 0;
    public static final int GPU1 = 1;
    public static final int GPU2 = 2;
    public static final int GPU3 = 3;
    public static final int GPU4 = 4;
    public static final int GPU5 = 5;
    public static final int GPU6 = 6;
    public static final int GPU7 = 7;
}
