package org.neuropod;

/**
 * This class is responsible for loading jni library
 */
public class LibraryLoader {
    private static boolean isLoaded_ = false;
    private static final String LIBNAME = "neuropod_jni";

    /**
     * Load jni library.
     */
    public static void load() {
        if (isLoaded_) {
            return;
        }
        System.out.println("java.library.path: " + System.getProperty("java.library.path"));
        try {
            // TODO: support jar file by weijiad
            System.loadLibrary(LIBNAME);
        } catch (UnsatisfiedLinkError e) {
            System.out.println("Load library failed " + e.getMessage());
        }
        isLoaded_ = true;
    }
}

