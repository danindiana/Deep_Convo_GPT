class GRU {
    constructor(inputSize, hiddenSize) {
        // Weights and biases
        this.Wz = createRandomMatrix(hiddenSize, inputSize + hiddenSize);
        this.Wr = createRandomMatrix(hiddenSize, inputSize + hiddenSize);
        this.Wh = createRandomMatrix(hiddenSize, inputSize + hiddenSize);
        this.bz = createRandomVector(hiddenSize);
        this.br = createRandomVector(hiddenSize);
        this.bh = createRandomVector(hiddenSize);

        // Hidden state
        this.h_prev = createRandomVector(hiddenSize);
    }

    forward(xt) {
        const z = concatenate(xt, this.h_prev);
        const zt = sigmoid(dot(this.Wz, z) + this.bz);
        const rt = sigmoid(dot(this.Wr, z) + this.br);
        const htHat = tanh(dot(this.Wh, concatenate(xt, multiply(rt, this.h_prev))) + this.bh);
        const ht = multiply(zt, this.h_prev) + multiply(subtract(1, zt), htHat);
        return { h_t: ht, h_prev: ht };
    }
}

// Helper functions
function createRandomMatrix(rows, cols) {
    return Array.from({ length: rows }, () =>
        Array.from({ length: cols }, () => Math.random())
    );
}

function createRandomVector(size) {
    return Array.from({ length: size }, () => Math.random());
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function tanh(x) {
    return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
}

function dot(a, b) {
    if (a.length !== b.length) {
        throw new Error("Vectors must have the same length for dot product");
    }
    return a.reduce((sum, x, i) => sum + x * b[i], 0);
}

function multiply(a, b) {
    if (a.length !== b.length) {
        throw new Error("Vectors must have the same length for element-wise multiplication");
    }
    return a.map((x, i) => x * b[i]);
}

function subtract(a, b) {
    if (a.length !== b.length) {
        throw new Error("Vectors must have the same length for subtraction");
    }
    return a.map((x, i) => x - b[i]);
}

function concatenate(a, b) {
    return [...a, ...b];
}
