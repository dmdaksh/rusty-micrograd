use rusty_micrograd::GraphArena;
use rusty_micrograd::mlp::{Layer, MLP, Neuron};

fn main() {
    let mut arena = GraphArena::<f32>::new();
    let x_ids = vec![arena.input(0.5_f32), arena.input(-1.2_f32)];
    let neuron = Neuron::new(vec![0.8_f32, -0.4_f32], 0.1_f32, GraphArena::tanh);

    let layer = Layer::new(vec![neuron]);
    let mut mlp = MLP::new(vec![layer]);
    let out_ids = mlp.forward(&mut arena, &x_ids);
    arena.backward(out_ids[0]);
    arena.print_graph(out_ids[0]);
}
