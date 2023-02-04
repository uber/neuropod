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
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * This class is responsible for loading jni library
 *
 * <p>The Java bindings require a native (JNI) library. This library
 * (libneuropod_jni.so on Linux, libneuropod_jni.dylib on OS X)
 * can be made available to the JVM using the java.library.path System property (e.g., using
 * -Djava.library.path command-line argument).
 *
 * <p>Alternatively, the native libraries can be packed in a .jar.
 * However, in such cases, the native library has to be extracted from the .jar archive.
 *
 * <p>LibraryLoader.load() takes care of this. First looking for the library in java.library.path,
 * if failed it tries to find the OS and architecture specific version of the library in the
 * set of ClassLoader resources (under com/uber/neuropod/native/OS-ARCH).
 */
final class LibraryLoader {
    private LibraryLoader() {
    }

    private static final Logger LOGGER = Logger.getLogger(LibraryLoader.class.getName());
    // C++ library names. If we use the Neruopod Java API from a jar file these files may be packed into the jar file.
    // Note that embedded native files are OS-Arch specific and may be absent for curent platform.
    private static final List<String> EMBEDDED_LIB_NAMES = Arrays.asList("libneuropod.so");
    private static final List<String> BIN_NAMES = Arrays.asList("neuropod_multiprocess_worker");

    private static final String NEUROPOD_BASE_DIR_ENV_VAR = "NEUROPOD_BASE_DIR";
    private static final String NEUROPOD_BASE_SUBDIR = "neuropod_jni_backends";
    private static final String TENSORFLOW_BACKEND = "neuropod_tensorflow_backend.tar.gz";
    private static final String TORCHSCRIPT_BACKEND = "neuropod_torchscript_backend.tar.gz";
    private static final List<String> EMBEDDED_BACKEND_PACKAGES_NAMES = Arrays.asList(TENSORFLOW_BACKEND, TORCHSCRIPT_BACKEND);

    // System.loadLibrary can autocompele the libname.
    private static final String JNI_NAME = "neuropod_jni";
    private static final int BUFFER_SIZE = 1 << 20;
    private static final String TEMP_DIRECTORY_NAME = "neuropod_native_libraries";

    /**
     * Load jni library.
     */
    public static void load() {
        // Java static initializers are thread safe.
        if (isLoaded() || loadSharedLibrary() || loadEmbeddedLibrary()) {
            // Either:
            // (1) The native library has already been statically loaded, OR
            // (2) The required native code has been statically linked (through a custom launcher), OR
            // (3) The native code is part of another library.
            // (4) It has been packaged into the .jar file and loaded by loadEmbeddedLibrary.
            //
            // Somehow the native code is loaded, so nothing else to do.
            return;
        }
        throw new UnsatisfiedLinkError("load " + System.mapLibraryName(JNI_NAME) + " failed");
    }

    /**
     * Load the jni library from the system library path (-Djava.library.path)
     */
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

    private static String makeResourcePath() {
        return "/com/uber/neuropod/native/" + String.format("%s-%s/", os(), architecture());
    }

    /**
     * Load the jni library, neuropod core and backend from the jar file.
     */
    private static boolean loadEmbeddedLibrary() {
        if (os().equals("unsupported")) {
            throw new NeuropodJNIException("unsupported OS");
        }
        try {
            final Path tempPath = Files.createTempDirectory(TEMP_DIRECTORY_NAME).toAbsolutePath();
            tempPath.toFile().deleteOnExit();
            String libAbsPath = tempPath.toString();
            String resPath = makeResourcePath();

            for (String libName : EMBEDDED_LIB_NAMES) {
                File embeddedLibFile = extractFile(libAbsPath, resPath, libName);
                if (embeddedLibFile != null) {
                    LOGGER.log(Level.INFO, String.format("Extracted embedded lib file %s", embeddedLibFile));
                }
            }

            for (String binName : BIN_NAMES) {
                File binFile = extractFile(libAbsPath, resPath, binName);
                if (binFile != null) {
                    LOGGER.log(Level.INFO, String.format("Extracted bin file %s", binFile));
                    boolean succeeded = binFile.setExecutable(true);
                    LOGGER.log(Level.INFO, String.format("Set everybody's execute permission %s", succeeded));
                }
            }

            File libFile = extractFile(libAbsPath, resPath, System.mapLibraryName(JNI_NAME));
            if (libFile != null) {
                System.load(libFile.getCanonicalPath());
            }

            nativeExport(libAbsPath);

            // Native Neuropod Core is using environment variable NEUROPOD_BASE_DIR if it is set,
            // to register neuropod backends.
            // otherwise it uses path to system local libraries (/usr/local/lib/neuropod).

            // Resources of this package may have embedded backend packages packed as tar (optional).
            // Neuropod Java API library uses strategy that uses setup of existing "neuropod" environments
            // and allows fat jar to configure "non-neuropod" environment.
            // Existing "neuropod" system is detected by envvar NEUROPOD_BASE_DIR that should have a path
            // to pre-installed neuropod backends.

            String neuropodBaseDir = System.getenv(NEUROPOD_BASE_DIR_ENV_VAR);
            if (neuropodBaseDir != null) {
                LOGGER.log(Level.INFO, String.format("Found NEUROPOD_BASE_DIR=%s", neuropodBaseDir));
                return true;
            }

            // If NEUROPOD_BASE_DIR is not set and Jar has backend packages, unpack and install it to
            // temp directory and set NEUROPOD_BASE_DIR that Neuropod Core register these backends.
            // Note that packages are tar.gz files that has "neuropod standard" directory structure, for instance:
            // 0.2.0/backends/tensorflow_1.15.0/
            // 0.2.0/backends/torchscript_1.4.0/
            // "register backends" looks at NEUROPOD_BASE_DIR and register backends by type and version.
            neuropodBaseDir = tempPath.resolve(NEUROPOD_BASE_SUBDIR).toString();

            final Path neuropodBasePath = Paths.get(neuropodBaseDir);
            LOGGER.log(Level.INFO, String.format("Create temp neuropod base directory %s", neuropodBasePath));
            Path directories = Files.createDirectories(neuropodBasePath.toAbsolutePath());

            boolean extracted = false;
            for (String libName : EMBEDDED_BACKEND_PACKAGES_NAMES) {
                File embeddedPackageFile = extractFile(libAbsPath, resPath, libName);
                if (embeddedPackageFile != null) {
                    extracted = true;
                    LOGGER.log(Level.INFO, String.format("Extracted embedded package file %s", embeddedPackageFile));
                    installFramework(libAbsPath, embeddedPackageFile.getName(), neuropodBaseDir);
                }
            }

            if (extracted) {
                LOGGER.log(Level.INFO, String.format("Set NEUROPOD_BASE_DIR=%s", neuropodBaseDir));
                long res = nativeSetEnv(NEUROPOD_BASE_DIR_ENV_VAR, neuropodBaseDir);
            }

        } catch (IOException e) {
            LOGGER.log(Level.WARNING, e.getMessage());
            return false;
        }
        return true;
    }

    private static boolean installFramework(String libAbsPath, String frameworkPackage, String neuropodBaseDir) {
        return startShellCommand(String.format("tar -xf %s/%s -C %s", libAbsPath, frameworkPackage, neuropodBaseDir));
    }

    private static boolean startShellCommand(String cmd) {
        LOGGER.log(Level.INFO, String.format("Command: %s``", cmd));
        try {
            ProcessBuilder builder = new ProcessBuilder();
            builder.command("sh", "-c", cmd);
            builder.directory(new File(System.getProperty("user.home")));
            Process process = builder.start();
            int exitCode = process.waitFor();
            LOGGER.log(Level.INFO, String.format("Exit code %s", exitCode));
        } catch (IOException e) {
            LOGGER.log(Level.WARNING, e.getMessage());
            return false;
        } catch(InterruptedException e) {
            LOGGER.log(Level.WARNING, e.getMessage());
            return false;
        }
        return true;
    }

    private static File extractFile(String targetDir, String resPath, String localLibName) throws IOException {
        URL nativeLibraryUrl = LibraryLoader.class.getResource(resPath + localLibName);
        if (nativeLibraryUrl == null) {
            LOGGER.log(Level.WARNING, "File %s not found in the jar package, should be fine if it is part " +
                    "of a backend library that is not packed into the jar file", resPath + localLibName);
            return null;
        }
        final File targetFile = new File(targetDir, localLibName);
        targetFile.deleteOnExit();
        final InputStream in = nativeLibraryUrl.openStream();
        final OutputStream out = new BufferedOutputStream(new FileOutputStream(targetFile));
        // Copy library file in jar to temporary file.
        int len;
        byte[] buffer = new byte[BUFFER_SIZE];
        while ((len = in.read(buffer)) > -1)
            out.write(buffer, 0, len);
        out.close();
        in.close();
        return targetFile;
    }

    private static String os() {
        final String p = System.getProperty("os.name").toLowerCase();
        if (p.contains("linux")) {
            return "linux";
        } else if (p.contains("os x") || p.contains("darwin")) {
            return "darwin";
        } else {
            return "unsupported";
        }
    }

    private static String architecture() {
        final String arch = System.getProperty("os.arch").toLowerCase();
        return (arch.equals("amd64")) ? "x86_64" : arch;
    }

    private static boolean isLoaded() {
        try {
            // Return value isn't important, it throws if not loaded.
            boolean loaded = nativeIsLoaded();
            LOGGER.log(Level.INFO, String.format("LibraryLoader.isLoaded %s", loaded));
            return true;
        } catch (UnsatisfiedLinkError e) {
            return false;
        }
    }

    private static native boolean nativeIsLoaded();

    private static native void nativeExport(String path);

    private static native long nativeSetEnv(String name, String value);
}
