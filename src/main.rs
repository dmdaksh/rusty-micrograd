use rusty_micrograd::Value;

fn main() {
    let a = Value::new(3.0);
    a.set_label("a");
    println!("a: {}", a);
    let b = Value::new(4.0);
    b.set_label("b");
    println!("b: {}", b);
    let c = &a + &b;
    c.set_label("c");
    println!("c: {}", c);

    let d = Value::new(2.0);
    d.set_label("d");
    println!("d: {}", d);
    let e = Value::new(5.0);
    e.set_label("e");
    println!("e: {}", e);
    let f = &d * &e;
    f.set_label("f");
    println!("f: {}", f);

    let g = &c + &f;
    g.set_label("g");
    println!("g: {}", g);

    g.print_graph();
}
