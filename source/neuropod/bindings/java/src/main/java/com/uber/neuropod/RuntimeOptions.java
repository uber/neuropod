package com.uber.neuropod;

/**
 * An One to One mapping to RuntimeOptions native class.
 */
public class RuntimeOptions {

    /**
     * Is use ope.
     *
     * Whether or not to use out-of-process execution
     * (using shared memory to communicate between the processes)
     */
    public boolean useOpe = false;

    /**
     * Is free memory every cycle.
     *
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
     *
     * This option can be used to run the neuropod in an existing worker process
     * If this string is empty, a new worker will be started.
     */
    public String controlQueueName = "";

    /**
     * The visible device.
     *
     * Some devices are defined in the NeuropodDevice class. For machines with more
     * than 8 GPUs, passing in an index will also work (e.g. `9` for `GPU9`).
     * To attempt to run the model on CPU, set this to `NeuropodDevice.CPU`
     */
    public int visibleDevice = NeuropodDevice.GPU0;

    /**
     * Is load model at construction time.
     *
     * Sometimes, it's important to be able to instantiate a Neuropod without
     * immediately loading the model. If this is set to `false`, the model will
     * not be loaded until the `loadModel` method is called on the Neuropod.
     */
    public boolean loadModelAtConstruction = true;

    /**
     * Is disable shape and type checking.
     *
     * disableShapeAndTypeCheckingBoolean means whether or not to disable shape
     * and type checking when running inference.
     */
    public boolean disableShapeAndTypeChecking = false;

    /**
     * Instantiates a new Runtime options.
     */
    public RuntimeOptions() {}

}
