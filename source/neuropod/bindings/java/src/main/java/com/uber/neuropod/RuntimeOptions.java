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
 * An One to One mapping to RuntimeOptions native class.
 */
public class RuntimeOptions {

    /**
     * Is use ope.
     * <p>
     * Whether or not to use out-of-process execution
     * (using shared memory to communicate between the processes)
     */
    public boolean useOpe = false;

    /**
     * Is free memory every cycle.
     * <p>
     * Internally, OPE(out-of-process execution) uses a shared memory allocator that reuses
     * blocks of memory if possible. Therefore memory isn't necessarily allocated during each
     * inference cycle as blocks may be reused.
     * If freeMemoryEveryCycle is set, then unused shared memory will be freed every cycle
     * This is useful for simple inference, but for code that is pipelined
     * (e.g. generating inputs for cycle t + 1 during the inference of cycle t), this may not
     * be desirable.
     * If freeMemoryEveryCycle is false, the user is responsible for periodically calling
     * Neuropod.freeUnusedShmBlocks()
     */
    public boolean freeMemoryEveryCycle = true;


    /**
     * The control queue name.
     * <p>
     * This option can be used to run the neuropod in an existing worker process
     * If this string is empty, a new worker will be started.
     */
    public String controlQueueName = "";

    /**
     * The visible device.
     * <p>
     * Some devices are defined in the NeuropodDevice class. For machines with more
     * than 8 GPUs, passing in an index will also work (e.g. `9` for `GPU9`).
     * To attempt to run the model on CPU, set this to `NeuropodDevice.CPU`
     */
    public int visibleDevice = NeuropodDevice.GPU0;

    /**
     * Is load model at construction time.
     * <p>
     * Sometimes, it's important to be able to instantiate a Neuropod without
     * immediately loading the model. If this is set to `false`, the model will
     * not be loaded until the `loadModel` method is called on the Neuropod.
     */
    public boolean loadModelAtConstruction = true;

    /**
     * Is disable shape and type checking.
     * <p>
     * disableShapeAndTypeCheckingBoolean means whether or not to disable shape
     * and type checking when running inference.
     */
    public boolean disableShapeAndTypeChecking = false;

    /**
     * Instantiates a new Runtime options.
     */
    public RuntimeOptions() {
    }

    /**
     * To native runtime options native.
     *
     * @return the native runtime options
     */
    RuntimeOptionsNative toNative() {
        return new RuntimeOptionsNative(useOpe,
                freeMemoryEveryCycle,
                controlQueueName,
                visibleDevice,
                loadModelAtConstruction,
                disableShapeAndTypeChecking);
    }

    /**
     * The corresponding native type for RuntimeOptions. Is not exposed to user, only used inside other public
     * method.
     */
    static class RuntimeOptionsNative extends NativeClass {
        /**
         * Instantiates a new Runtime options native.
         *
         * @param useOpe                      use ope
         * @param freeMemoryEveryCycle        the free memory every cycle
         * @param controlQueueName            the control queue name
         * @param visibleDevice               the visible device
         * @param loadModelAtConstruction     the load model at construction
         * @param disableShapeAndTypeChecking the disable shape and type checking
         */
        RuntimeOptionsNative(boolean useOpe,
                             boolean freeMemoryEveryCycle,
                             String controlQueueName,
                             int visibleDevice,
                             boolean loadModelAtConstruction,
                             boolean disableShapeAndTypeChecking) {
            super(nativeCreate(useOpe,
                    freeMemoryEveryCycle,
                    controlQueueName,
                    visibleDevice,
                    loadModelAtConstruction,
                    disableShapeAndTypeChecking));
        }

        static private native long nativeCreate(boolean useOpe,
                                                boolean freeMemoeryEverySycle,
                                                String controlQueueName,
                                                int visibleDevice,
                                                boolean loadModelAtConstruction,
                                                boolean disableShapeAndTypeChecking);

        @Override
        protected native void nativeDelete(long handle);
    }

}
