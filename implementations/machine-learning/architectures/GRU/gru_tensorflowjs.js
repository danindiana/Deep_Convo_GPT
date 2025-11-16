import * as tf from '@tensorflow/tfjs';

class GRU {
    constructor(inputSize, hiddenSize) {
        // Weights and biases (initialized as trainable variables)
        this.Wz = tf.variable(tf.randomNormal([hiddenSize, inputSize + hiddenSize]));
        this.Wr = tf.variable(tf.randomNormal([hiddenSize, inputSize + hiddenSize]));
        this.Wh = tf.variable(tf.randomNormal([hiddenSize, inputSize + hiddenSize]));
        this.bz = tf.variable(tf.zeros([hiddenSize]));
        this.br = tf.variable(tf.zeros([hiddenSize]));
        this.bh = tf.variable(tf.zeros([hiddenSize]));

        // Hidden state (initialized with zeros)
        this.h_prev = tf.zeros([hiddenSize]);
    }

    forward(xt) {
        const z = tf.concat([xt, this.h_prev], 1);  // Concatenate input and previous hidden state
        const zt = tf.sigmoid(tf.add(tf.matMul(this.Wz, z), this.bz));
        const rt = tf.sigmoid(tf.add(tf.matMul(this.Wr, z), this.br));
        const htHat = tf.tanh(tf.add(tf.matMul(this.Wh, tf.concat([xt, tf.multiply(rt, this.h_prev)], 1)), this.bh));
        const ht = tf.add(tf.multiply(zt, this.h_prev), tf.multiply(tf.subtract(1, zt), htHat));
        return { h_t: ht, h_prev: ht };
    }
}
