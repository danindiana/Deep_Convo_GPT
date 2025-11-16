// File: brain_model.rs
use rand::Rng;
use std::sync::Arc;
use std::sync::Mutex;

pub struct Neuron {
    weights: Vec<f32>,
}

impl Neuron {
    pub fn new(connections: usize) -> Neuron {
        let mut rng = rand::thread_rng();
        let mut weights = Vec::new();
        for _ in 0..connections {
            weights.push(rng.gen());
        }
        Neuron { weights }
    }

    pub fn process(&self, inputs: &Vec<f32>) -> f32 {
        self.weights.iter().zip(inputs).map(|(w, i)| w * i).sum()
    }
}

pub struct Brain {
    neurons: Arc<Mutex<Vec<Neuron>>>,
}

impl Brain {
    pub fn new(size: usize) -> Brain {
        let mut neurons = Vec::new();
        for _ in 0..size {
            neurons.push(Neuron::new(size));
        }
        Brain {
            neurons: Arc::new(Mutex::new(neurons)),
        }
    }

    pub fn encode(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .lock()
            .unwrap()
            .iter()
            .map(|neuron| neuron.process(&inputs))
            .collect()
    }

    pub fn retrieve(&self, encoded: Vec<f32>) -> Vec<f32> {
        self.neurons
            .lock()
            .unwrap()
            .iter()
            .map(|neuron| neuron.process(&encoded))
            .collect()
    }
}

fn main() {
    let brain = Brain::new(100);
    let sensory_input = vec![0.5; 100];
    let encoded = brain.encode(sensory_input.clone());
    let retrieved = brain.retrieve(encoded);
    println!("Sensory input: {:?}", sensory_input);
    println!("Encoded: {:?}", encoded);
    println!("Retrieved: {:?}", retrieved);
}
