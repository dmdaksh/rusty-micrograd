use num_traits::Float;
use std::collections::HashSet;
use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Mul, MulAssign},
    rc::{Rc, Weak},
};

#[derive(Clone, Debug)]
struct Inner<T: Float + Copy> {
    data: T,
    grad: T,
    oper: char,
    label: String,
    prev: Vec<Weak<RefCell<Inner<T>>>>,
}

#[derive(Clone, Debug)]
pub struct Value<T: Float + Copy> {
    inner: Rc<RefCell<Inner<T>>>,
}

impl<T> Value<T>
where
    T: Float + Copy,
{
    pub fn new(data: T) -> Self {
        Value {
            inner: Rc::new(RefCell::new(Inner {
                data,
                grad: T::zero(),
                oper: '\0',
                label: String::new(),
                prev: Vec::new(),
            })),
        }
    }

    pub fn new_with_oper(data: T, oper: char) -> Self {
        Value {
            inner: Rc::new(RefCell::new(Inner {
                data,
                grad: T::zero(),
                oper,
                label: String::new(),
                prev: Vec::new(),
            })),
        }
    }

    pub fn get(&self) -> T {
        self.inner.borrow().data
    }

    pub fn grad(&self) -> T {
        self.inner.borrow().grad
    }

    pub fn set_data(&mut self, data: T) {
        self.inner.borrow_mut().data = data;
    }

    pub fn set_label(&self, label: impl Into<String>) {
        self.inner.borrow_mut().label = label.into();
    }

    pub fn add_prev(&self, child: &Value<T>) {
        let weak_child = Rc::downgrade(&child.inner);
        self.inner.borrow_mut().prev.push(weak_child);
    }
}

impl<T> Display for Value<T>
where
    T: Float + Copy + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // print data and data of previous values
        let prev_data: Vec<String> = self
            .inner
            .borrow()
            .prev
            .iter()
            .filter_map(|weak| weak.upgrade())
            .map(|rc| rc.borrow().data.to_string())
            .collect();

        if self.inner.borrow().oper == '\0' {
            write!(
                f,
                "Value(data: {}) [prev: {}]",
                self.inner.borrow().data,
                if prev_data.is_empty() {
                    "none".to_string()
                } else {
                    prev_data.join(", ")
                }
            )
        } else {
            write!(
                f,
                "Value(data: {}, oper: '{}') [prev: {}]",
                self.inner.borrow().data,
                self.inner.borrow().oper,
                if prev_data.is_empty() {
                    "none".to_string()
                } else {
                    prev_data.join(", ")
                }
            )
        }
    }
}

impl<T> Add<Value<T>> for Value<T>
where
    T: Float + Copy + Add<Output = T>,
{
    type Output = Value<T>;

    fn add(self, other: Value<T>) -> Self::Output {
        (&self).add(&other)
    }
}

impl<'a, 'b, T> Add<&'b Value<T>> for &'a Value<T>
where
    T: Float + Copy + Add<Output = T>,
{
    type Output = Value<T>;

    fn add(self, other: &'b Value<T>) -> Value<T> {
        let new_data = self.get() + other.get();
        let result = Value::new_with_oper(new_data, '+');
        result.add_prev(self);
        result.add_prev(other);
        result
    }
}

impl<T> Mul<Value<T>> for Value<T>
where
    T: Float + Copy + Mul<Output = T>,
{
    type Output = Value<T>;

    fn mul(self, other: Value<T>) -> Self::Output {
        (&self).mul(&other)
    }
}

impl<'a, 'b, T> Mul<&'b Value<T>> for &'a Value<T>
where
    T: Float + Copy + Mul<Output = T>,
{
    type Output = Value<T>;

    fn mul(self, other: &'b Value<T>) -> Value<T> {
        let new_data = self.get() * other.get();
        let result = Value::new_with_oper(new_data, '*');
        result.add_prev(self);
        result.add_prev(other);
        result
    }
}

impl<T> AddAssign<Value<T>> for Value<T>
where
    T: Float + Copy + AddAssign,
{
    fn add_assign(&mut self, other: Value<T>) {
        self.add_assign(&other);
    }
}

impl<'a, T> AddAssign<&'a Value<T>> for Value<T>
where
    T: Float + Copy + AddAssign,
{
    fn add_assign(&mut self, other: &'a Value<T>) {
        let new_data = self.get() + other.get();
        self.set_data(new_data);
        self.add_prev(other);
    }
}

impl<T> MulAssign<Value<T>> for Value<T>
where
    T: Float + Copy + MulAssign,
{
    fn mul_assign(&mut self, other: Value<T>) {
        self.mul_assign(&other);
    }
}

impl<'a, T> MulAssign<&'a Value<T>> for Value<T>
where
    T: Float + Copy + MulAssign,
{
    fn mul_assign(&mut self, other: &'a Value<T>) {
        let new_data = self.get() * other.get();
        self.set_data(new_data);
        self.add_prev(other);
    }
}

impl<T> Value<T>
where
    T: Float + Copy + Display,
{
    /// Pretty-print the computation graph from here down.
    pub fn print_graph(&self) {
        let mut visited = HashSet::new();
        self.print_node(&[], &mut visited);
    }

    fn print_node(&self, ancestors_last: &[bool], visited: &mut HashSet<*const RefCell<Inner<T>>>) {
        // cycle check
        let ptr = Rc::as_ptr(&self.inner);
        if !visited.insert(ptr) {
            return;
        }

        // draw prefixes
        for &was_last in ancestors_last {
            print!("{}", if was_last { "    " } else { "|   " });
        }

        // branch glyph
        if !ancestors_last.is_empty() {
            let is_last = *ancestors_last.last().unwrap();
            print!("{}", if is_last { "|__ " } else { "|-- " });
        }

        // decide what to show
        let inner = self.inner.borrow();
        let line = if !inner.label.is_empty() {
            if inner.oper == '\0' {
                format!("{}: {}", inner.label, inner.data)
            } else {
                format!("{}: {} ({})", inner.label, inner.data, inner.oper)
            }
        } else if inner.oper != '\0' {
            format!("{} ({})", inner.data, inner.oper)
        } else {
            inner.data.to_string()
        };
        println!("{}", line);
        drop(inner);

        // recurse into children
        let children: Vec<_> = self
            .inner
            .borrow()
            .prev
            .iter()
            .filter_map(Weak::upgrade)
            .collect();
        let n = children.len();
        for (i, child_rc) in children.iter().enumerate() {
            let mut flags = ancestors_last.to_vec();
            flags.push(i == n - 1);
            let child = Value {
                inner: child_rc.clone(),
            };
            child.print_node(&flags, visited);
        }
    }
}
