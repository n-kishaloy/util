use util;

fn main() {
    println!("{:.25}", util::g_search(|x| (x - 4.5).powf(2.000) + 5.0, 80.2, -10.4, 1e-10).unwrap().0);
}