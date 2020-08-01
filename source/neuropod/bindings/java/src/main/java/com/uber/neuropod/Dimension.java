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

/**
 * The type Dimension.
 */
public class Dimension {
    /**
     * The Value. -1 means None/null, any value is OK
     * -2 means this is a symbol (see below)
     */
    long value;
    /**
     * The name of this symbol (if it is a symbol).
     */
    String symbol;

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
}
