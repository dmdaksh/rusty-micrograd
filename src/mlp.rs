use crate::arena::GraphArena;
use num_traits::Float;

/// A high-level Module trait: anything that can forward through the graph.
pub trait Module<T: Float + Copy> {
    fn forward(&mut self, arena: &mut GraphArena<T>, inputs: &[usize]) -> Vec<usize>;
}

/// Activation function type that operates on node IDs within the graph.
pub type Activation<T> = fn(&mut GraphArena<T>, usize) -> usize;

/// A single neuron: weighted sum + bias + activation via graph operations.
pub struct Neuron<T: Float + Copy> {
    pub weights: Vec<T>,
    pub bias: T,
    pub activation: Activation<T>,
}

impl<T: Float + Copy> Neuron<T> {
    pub fn new(weights: Vec<T>, bias: T, activation: Activation<T>) -> Self {
        Neuron {
            weights,
            bias,
            activation,
        }
    }
}

impl<T: Float + Copy> Module<T> for Neuron<T> {
    fn forward(&mut self, arena: &mut GraphArena<T>, inputs: &[usize]) -> Vec<usize> {
        // weighted sum node
        let mut sum_id = arena.input(self.bias);
        for (&inp, &w) in inputs.iter().zip(self.weights.iter()) {
            let w_id = arena.input(w);
            let prod_id = arena.mul(inp, w_id);
            sum_id = arena.add(sum_id, prod_id);
        }
        // apply activation operation in graph
        let out_id = (self.activation)(arena, sum_id);
        vec![out_id]
    }
}

/// A layer: a collection of neurons.
pub struct Layer<T: Float + Copy> {
    pub neurons: Vec<Neuron<T>>,
}

impl<T: Float + Copy> Layer<T> {
    pub fn new(neurons: Vec<Neuron<T>>) -> Self {
        Layer { neurons }
    }
}

impl<T: Float + Copy> Module<T> for Layer<T> {
    fn forward(&mut self, arena: &mut GraphArena<T>, inputs: &[usize]) -> Vec<usize> {
        self.neurons
            .iter_mut()
            .flat_map(|n| n.forward(arena, inputs))
            .collect()
    }
}

/// A multi-layer perceptron: sequence of layers.
pub struct MLP<T: Float + Copy> {
    pub layers: Vec<Layer<T>>,
}

impl<T: Float + Copy> MLP<T> {
    pub fn new(layers: Vec<Layer<T>>) -> Self {
        MLP { layers }
    }
    pub fn forward(&mut self, arena: &mut GraphArena<T>, inputs: &[usize]) -> Vec<usize> {
        let mut out = inputs.to_vec();
        for layer in &mut self.layers {
            out = layer.forward(arena, &out);
        }
        out
    }
}
