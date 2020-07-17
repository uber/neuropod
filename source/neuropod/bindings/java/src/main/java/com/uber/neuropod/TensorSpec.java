package com.uber.neuropod;

import java.util.List;

/**
 * The type Tensor spec.
 */
public class TensorSpec {
    private String name;
    private TensorType type;
    private List<Dimension> dims;

    /**
     * Instantiates a new Tensor spec.
     *
     * @param name the name
     * @param type the type
     * @param dims the dims
     */
    public TensorSpec(String name, TensorType type, List<Dimension> dims) {
        this.name = name;
        this.type = type;
        this.dims = dims;
    }

    /**
     * Gets name.
     *
     * @return the name
     */
    public String getName() {
        return name;
    }

    /**
     * Gets type.
     *
     * @return the type
     */
    public TensorType getType() {
        return type;
    }

    /**
     * Gets dims.
     *
     * @return the dims
     */
    public List<Dimension> getDims() {
        return dims;
    }
}
