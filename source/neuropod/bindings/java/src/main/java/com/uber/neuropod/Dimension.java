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
