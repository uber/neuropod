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

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.net.URL;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * This class is responsible for loading jni library
 */
class LibraryLoader {
    private LibraryLoader() {
    }

    private static final Logger LOGGER = Logger.getLogger(LibraryLoader.class.getName());
    // C++ library names. If we use the Neruopod Java API from a jar file these files may be packed into the jar file.
    private static final List<String> EMBEDDED_LIB_NAMES = Arrays.asList("libneuropod.so");
    private static final List<String> BIN_NAMES = Arrays.asList("neuropod_multiprocess_worker");
    private static final String JNI_NAME = "neuropod_jni"; // System.loadLibrary can autocompele the libname
    private static final int BUFFER_SIZE = 1 << 20;
    private static final String TEMP_DIRECTORY_NAME = "neuropod_native_libraries";

    /**
     * Load jni library.
     */
    public static void load() {
        // Java static initializers are thread safe
        if (isLoaded() || loadSharedLibrary() || loadEmbeddedLibrary()) {
            return;
        }
        throw new UnsatisfiedLinkError("load " + System.mapLibraryName(JNI_NAME) + " failed");
    }

    // Load the jni library from the system library path
    private static boolean loadSharedLibrary() {
        try {
            System.loadLibrary(JNI_NAME);
            return true;
        } catch (UnsatisfiedLinkError e) {
            LOGGER.log(Level.WARNING, "try loading shared library failed, should be fine if using the " +
                    "library from a jar file: " + e.getMessage());
            return false;
        }
    }

    // Load the jni library from the jar file
    private static boolean loadEmbeddedLibrary() {
        // TODO(weijiad): Should be /com/uber/neuropod/native/{OS}/{PLATFORM} once we figure out how to copy native
        // libraries to that path.
        String resPath = "/";
        if (getOS().equals("unsupported")) {
            throw new NeuropodJNIException("unsupported OS");
        }
        try {
            final File tempPath = Files.createTempDirectory(TEMP_DIRECTORY_NAME).toFile();
            tempPath.deleteOnExit();
            String libAbsPath = tempPath.getCanonicalPath().toString();
            for (String libName : EMBEDDED_LIB_NAMES) {
                File libFile = extractFile(libAbsPath, resPath, libName);
            }
            for (String binName : BIN_NAMES) {
                File binFile = extractFile(libAbsPath, resPath, binName);
                if (binFile != null) {
                    binFile.setExecutable(true);
                }
            }
            File libFile = extractFile(libAbsPath, resPath, System.mapLibraryName(JNI_NAME));
            System.load(libFile.getCanonicalPath());
            nativeExport(libAbsPath);
        } catch (IOException e) {
            System.out.println(e.getMessage());
            return false;
        }
        return true;
    }

    private static File extractFile(String targetDir, String resPath, String localLibName) throws IOException {
        URL nativeLibraryUrl = LibraryLoader.class.getResource(resPath + localLibName);
        if (nativeLibraryUrl == null) {
            LOGGER.log(Level.WARNING, "File {0} not found in the jar package, should be fine if it is part " +
                    "of a backend library that is not packed into the jar file", resPath + localLibName);
            return null;
        }
        final File targetFile = new File(targetDir, localLibName);
        targetFile.deleteOnExit();
        final InputStream in = nativeLibraryUrl.openStream();
        final OutputStream out = new BufferedOutputStream(new FileOutputStream(targetFile));
        // Copy library file in jar to temporary file
        int len = 0;
        byte[] buffer = new byte[BUFFER_SIZE];
        while ((len = in.read(buffer)) > -1)
            out.write(buffer, 0, len);
        out.close();
        in.close();
        return targetFile;
    }

    private static String getOS() {
        final String p = System.getProperty("os.name").toLowerCase();
        if (p.contains("linux")) {
            return "linux";
        } else if (p.contains("os x") || p.contains("darwin")) {
            return "darwin";
        } else {
            return "unsupported";
        }
    }


    private static boolean isLoaded() {
        try {
            return nativeIsLoaded();
        } catch (UnsatisfiedLinkError e) {
            return false;
        }
    }

    private static native boolean nativeIsLoaded();

    private static native void nativeExport(String path);
}
