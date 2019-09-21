extern crate ndarray;
extern crate itertools;

use itertools::izip;
use util;
use ndarray::{array, Array1, ArrayView1};



fn main() {
    println!("{:?}", util::optim::g_search(|x| (x - 4.5).powf(2.000) + 5.0, 5.2, -1.0, 1e-10).unwrap().0 - 4.5);
    println!("{:?}", util::optim::g_search(|x| (x - 4.5).powf(2.000) + 5.0, -10.2, 80.4, 1e-10).unwrap().0 - 4.5);

    println!("pon => {:?}", util::optim::conj_grad_pr(
            |x|     (x[0]-3.0).powf(4.0) 
                    + (x[1]-4.0).powf(2.0) * (x[2]-2.0).powf(2.0) 
                    + (x[2]-2.0).powf(4.0) + 10.0 + (x[1]-4.0).powf(2.0),
                &array![3.2,4.2,1.8], 1e-6, 1e-8, 1e-12
        )   
    );

    let xa:Vec<i32> = vec![1,2,3,4,5,6];
    let xb:Vec<f64> = vec![2.0, 3.5, 2.9, 6.5, 7.5];
    let xc:Vec<i64> = vec![4,5,6,7];

    let xn:Vec<(&i32, &f64, &i64)> = 
        xa  .iter()
            .zip(&xb)
            .zip(&xc)
            .map(|((x, y), z)| (x,y,z) )
            .collect();
    
    println!("Zipper => {:?}", xn);

    let xp:Vec<(&i32,&f64,&i64,i32)> = izip!(&xa, &xb, &xc, 0..).collect();
    println!("IZipper => {:?}", xp);
    println!("xa      => {:?}", xa); 

    println!("{:?}", util::optim::conj_grad_pr(
            |x| (x[0]-3.5).powf(2.0) + (x[1]-2.5).powf(2.0),
                &array![2.5,3.5], 1e-6, 1e-8, 1e-12
        )   
    );




}