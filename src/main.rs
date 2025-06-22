use rusty_micrograd::GraphArena;

fn main() {
    let mut g = GraphArena::<f32>::new();
    let x = g.input(2.0);
    // let y = g.input(3.0);
    let z = g.mul(x, x);
    let w = g.add(z, x);
    // g.print_graph(w);
    g.backward(w);
    g.print_graph(w);
    // println!(
    //     "w data={}, x grad={}, y grad={}",
    //     g.nodes[w].data, g.nodes[x].grad, g.nodes[y].grad
    // );
}
