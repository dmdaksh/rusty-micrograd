use num_traits::Float;
use std::collections::HashSet;
use std::fmt::Display;

/// Operation type for each node in the graph.
#[derive(Debug)]
pub enum Op<T> {
    Input,
    Add,
    Sub,
    Mul,
    Div,
    Relu,
    Tanh,
    Pow(T),
}

/// A single node in the computation graph.
#[derive(Debug)]
pub struct Node<T: Float + Copy> {
    pub data: T,
    pub grad: T,
    pub op: Op<T>,
    pub parents: Vec<usize>,
}

/// Low-level arena holding nodes and topological order.
#[derive(Debug)]
pub struct GraphArena<T: Float + Copy> {
    pub nodes: Vec<Node<T>>,
    topo: Vec<usize>,
}

impl<T: Float + Copy> GraphArena<T> {
    /// Create a new, empty graph.
    pub fn new() -> Self {
        GraphArena {
            nodes: Vec::new(),
            topo: Vec::new(),
        }
    }

    /// Add an input (leaf) node.
    pub fn input(&mut self, data: T) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: T::zero(),
            op: Op::Input,
            parents: Vec::new(),
        });
        self.topo.push(idx);
        idx
    }

    /// Create a node representing addition: c = a + b.
    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let data = self.nodes[a].data + self.nodes[b].data;
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: T::zero(),
            op: Op::Add,
            parents: vec![a, b],
        });
        self.topo.push(idx);
        idx
    }

    /// Create a node representing subtraction: c = a - b.
    pub fn sub(&mut self, a: usize, b: usize) -> usize {
        let data = self.nodes[a].data - self.nodes[b].data;
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: T::zero(),
            op: Op::Sub,
            parents: vec![a, b],
        });
        self.topo.push(idx);
        idx
    }

    /// Create a node representing multiplication: c = a * b.
    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let data = self.nodes[a].data * self.nodes[b].data;
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: T::zero(),
            op: Op::Mul,
            parents: vec![a, b],
        });
        self.topo.push(idx);
        idx
    }

    /// Create a node representing division: c = a / b.
    pub fn div(&mut self, a: usize, b: usize) -> usize {
        let data = self.nodes[a].data / self.nodes[b].data;
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: T::zero(),
            op: Op::Div,
            parents: vec![a, b],
        });
        self.topo.push(idx);
        idx
    }

    /// ReLU activation: c = max(0, a).
    pub fn relu(&mut self, a: usize) -> usize {
        let x = self.nodes[a].data;
        let data = if x > T::zero() { x } else { T::zero() };
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: T::zero(),
            op: Op::Relu,
            parents: vec![a],
        });
        self.topo.push(idx);
        idx
    }

    /// Tanh activation: c = tanh(a).
    pub fn tanh(&mut self, a: usize) -> usize {
        let x = self.nodes[a].data;
        let data = x.tanh();
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: T::zero(),
            op: Op::Tanh,
            parents: vec![a],
        });
        self.topo.push(idx);
        idx
    }

    /// Power: c = a.powf(exponent).
    pub fn powf(&mut self, a: usize, exponent: T) -> usize {
        let data = self.nodes[a].data.powf(exponent);
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: T::zero(),
            op: Op::Pow(exponent),
            parents: vec![a],
        });
        self.topo.push(idx);
        idx
    }

    /// Perform backward pass from loss index to compute gradients.
    pub fn backward(&mut self, loss_idx: usize) {
        // Reset grads
        for node in &mut self.nodes {
            node.grad = T::zero();
        }
        // Seed gradient at loss
        self.nodes[loss_idx].grad = T::one();

        // Traverse in reverse topological order
        for &idx in self.topo.iter().rev() {
            let grad = self.nodes[idx].grad;
            let parents = &self.nodes[idx].parents;
            match self.nodes[idx].op {
                Op::Add => {
                    let [a, b] = <[usize; 2]>::try_from(parents.clone()).unwrap();
                    self.nodes[a].grad = self.nodes[a].grad + grad;
                    self.nodes[b].grad = self.nodes[b].grad + grad;
                }
                Op::Sub => {
                    let [a, b] = <[usize; 2]>::try_from(parents.clone()).unwrap();
                    self.nodes[a].grad = self.nodes[a].grad + grad;
                    self.nodes[b].grad = self.nodes[b].grad - grad;
                }
                Op::Mul => {
                    let [a, b] = <[usize; 2]>::try_from(parents.clone()).unwrap();
                    let da = self.nodes[b].data * grad;
                    let db = self.nodes[a].data * grad;
                    self.nodes[a].grad = self.nodes[a].grad + da;
                    self.nodes[b].grad = self.nodes[b].grad + db;
                }
                Op::Div => {
                    let [a, b] = <[usize; 2]>::try_from(parents.clone()).unwrap();
                    let da = grad / self.nodes[b].data;
                    let db =
                        -(self.nodes[a].data * grad) / (self.nodes[b].data * self.nodes[b].data);
                    self.nodes[a].grad = self.nodes[a].grad + da;
                    self.nodes[b].grad = self.nodes[b].grad + db;
                }
                Op::Relu => {
                    let a = parents[0];
                    let d = if self.nodes[a].data > T::zero() {
                        T::one()
                    } else {
                        T::zero()
                    };
                    self.nodes[a].grad = self.nodes[a].grad + d * grad;
                }
                Op::Tanh => {
                    let a = parents[0];
                    let y = self.nodes[idx].data;
                    let d = T::one() - y * y;
                    self.nodes[a].grad = self.nodes[a].grad + d * grad;
                }
                Op::Pow(exp) => {
                    let a = parents[0];
                    let x = self.nodes[a].data;
                    let d = exp * x.powf(exp - T::one());
                    self.nodes[a].grad = self.nodes[a].grad + d * grad;
                }
                Op::Input => {}
            }
        }
    }
}

impl<T: Float + Copy + Display> GraphArena<T> {
    /// Pretty-print the graph data and gradients from a given root index.
    pub fn print_graph(&self, root: usize) {
        let mut visited = HashSet::new();
        self.print_node(root, &[], &mut visited);
    }

    fn print_node(&self, idx: usize, ancestors_last: &[bool], visited: &mut HashSet<usize>) {
        if !visited.insert(idx) {
            return;
        }

        // Prefix lines
        for &is_last in ancestors_last {
            print!("{}", if is_last { "    " } else { "|   " });
        }
        if !ancestors_last.is_empty() {
            let last = *ancestors_last.last().unwrap();
            print!("{}", if last { "|__ " } else { "|-- " });
        }

        // Node details
        let node = &self.nodes[idx];
        let op_str = match node.op {
            Op::Add => "+",
            Op::Sub => "-",
            Op::Mul => "*",
            Op::Div => "/",
            Op::Relu => "relu",
            Op::Tanh => "tanh",
            Op::Pow(_) => "pow",
            Op::Input => "input",
        };
        println!(
            "{}: {:.4} ({}) [grad={:.4}]",
            idx, node.data, op_str, node.grad
        );

        // Recurse into parents
        let parents = &node.parents;
        let n = parents.len();
        for (i, &p) in parents.iter().enumerate() {
            let mut flags = ancestors_last.to_vec();
            flags.push(i == n - 1);
            self.print_node(p, &flags, visited);
        }
    }
}
