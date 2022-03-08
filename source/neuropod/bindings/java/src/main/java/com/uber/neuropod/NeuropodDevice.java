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

/**
 * Which device the model uses. Use const int instead of enum to
 * support number greater than 7, or namely GPU8 or more GPUs
 */
public class NeuropodDevice {
    private NeuropodDevice() {}
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
