use util;

fn main() {
    println!("{:?}", util::optim::g_search(|x| (x - 4.5).powf(2.000) + 5.0, 5.2, -1.0, 1e-10).unwrap().0 - 4.5);
    println!("{:?}", util::optim::g_search(|x| (x - 4.5).powf(2.000) + 5.0, -10.2, 80.4, 1e-10).unwrap().0 - 4.5);

    println!("{:?}", util::optim::conj_grad_PR(
            |x|     (x[0]-3.0).powf(4.0) 
                    + (x[1]-4.0).powf(2.0) * (x[2]-2.0).powf(2.0) 
                    + (x[2]-2.0).powf(4.0) + 10.0 + (x[1]-4.0).powf(2.0),
                &mut vec![3.2,4.2,1.8], 1e-6, 1e-8, 1e-12
        )   
    );


    // println!("{:?}", util::optim::conj_grad_PR(
    //         |x| (x[0]-3.5).powf(2.0) + (x[1]-2.5).powf(2.0),
    //             &mut vec![2.5,3.5], 1e-5
    //     )   
    // );




    println!("{:?}", util::explvec(&vec![1.2,2.3],&vec![2.4,3.5], 1.5));
}