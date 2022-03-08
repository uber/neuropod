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

import java.util.Objects;

/**
 * The type Dimension.
 */
public class Dimension {
    /**
     * The Value. -1 means None/null, any value is OK
     * -2 means this is a symbol (see below)
     */
    private long value;
    /**
     * The name of this symbol (if it is a symbol).
     */
    private String symbol;

    /**
     * Instantiates a new Dimension by given value
     *
     * @param value the value
     */
    public Dimension(long value) {
        this.value = value;
    }

    /**
     * Instantiates a new Dimension by given symbol
     *
     * @param symbol the symbol
     */
    public Dimension(String symbol) {
        this.symbol = symbol;
        this.value = -2;
    }

    /**
     * Gets the value.
     *
     * @return the value
     */
    public long getValue() {
        return value;
    }

    /**
     * Gets the symbol.
     *
     * @return the symbol
     */
    public String getSymbol() {
        return symbol;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Dimension)) return false;
        Dimension dimension = (Dimension) o;
        return getValue() == dimension.getValue() &&
                Objects.equals(getSymbol(), dimension.getSymbol());
    }

    @Override
    public int hashCode() {
        return Objects.hash(getValue(), getSymbol());
    }

    @Override
    public String toString() {
        if (value >= 0) {
            return String.valueOf(value);
        } else if (value == -1) {
            return "None";
        } else if (value == -2) {
            return symbol;
        }
        throw new NeuropodJNIException("Dimension object has unexpected value:" + value);
    }
}
