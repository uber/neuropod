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

import java.util.List;
import java.util.Objects;

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

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TensorSpec)) return false;
        TensorSpec that = (TensorSpec) o;
        return getName().equals(that.getName()) &&
                getType() == that.getType() &&
                getDims().equals(that.getDims());
    }

    @Override
    public int hashCode() {
        return Objects.hash(getName(), getType(), getDims());
    }

    @Override
    public String toString() {
        return "TensorSpec{" +
                "name='" + name + '\'' +
                ", type=" + type +
                ", dims=" + dims +
                '}';
    }
}
