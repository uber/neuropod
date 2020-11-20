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

import static org.junit.Assert.*;

public class LibraryLoaderTest {

    @org.junit.Test
    public void load() {
        try {
            LibraryLoader.load();
        } catch (Exception e) {
            System.out.println(e.getMessage());
            fail();
        }
    }

    @org.junit.Test
    public void loadUnsupported() {
        // Update os.name to unsupported os to test that this is detected.
        String current = System.getProperty("os.name");
        System.setProperty("os.name", "Windows");

        try {
            LibraryLoader.load();
            // It should throw exception and never reach this line.
            fail();
        } catch (Exception expected) {
            // Set property value back to avoid other failures.
            System.setProperty("os.name", current);
            assertTrue(expected.getMessage(), expected.getMessage().contains("unsupported OS"));
        }
    }
}
