pub mod engine;
pub use engine::Value;

pub mod arena;
pub use arena::GraphArena;

pub mod mlp;
pub use mlp::{Layer, MLP, Module, Neuron};

#[cfg(test)]
pub mod tests;
