# File: BrainModel.py
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Sensory input size
input_size = 100

# Initialize a single-layer neural network
brain_model = Sequential([
    Dense(1, input_dim=input_size, activation='sigmoid')
])

# Compile the model
brain_model.compile(optimizer='sgd', loss='binary_crossentropy')

# Function to encode information
def encode(sensory_input, attention):
    global brain_model

    # Update the model's weights based on the attention
    current_weights = brain_model.get_weights()
    new_weights = [weight * attention for weight in current_weights]
    brain_model.set_weights(new_weights)

    # Use the model to process the sensory input
    perception = brain_model.predict(sensory_input)

    return perception

# Function to retrieve information
def retrieve(memory_cue):
    global brain_model

    # Use the model to retrieve the memory
    memory = brain_model.predict(memory_cue)

    return memory
