extern crate ndarray;

use ndarray::{Array1, ArrayView1, array};

pub fn approx(x:f64, y:f64, xtol:f64)->bool { (x-y).abs() < xtol }

pub fn appr_vec(a:&Vec<f64>, b:&Vec<f64>, ztol:f64)->bool {
    a.iter().zip(b).fold(0.0,|s,(x,y)| s+(x-y).powf(2.0)) < ztol.powf(2.0)
}

pub fn approx_arr(a:ArrayView1<f64>, b:ArrayView1<f64>, ztol:f64)->bool {
    a.iter().zip(b).fold(0.0,|s,(x,y)| s+(x-y).powf(2.0)) < ztol.powf(2.0)
}

pub mod optim {
        
    use ndarray::{Array1, ArrayView1, array};

    pub fn g_search(f:impl Fn(f64)->f64,la:f64,lb:f64,xtol:f64)
            ->Result<(f64,f64), &'static str> {

        const GR:f64= 0.6180339887498948482045868343656381177203091798057628621354;
        let (mut a, mut b) = (la, lb);
        let (mut x1, mut x2) = (b - GR*(b-a), a + GR*(b-a));
        let (mut f1, mut f2) = ( f(x1), f(x2) );

        if f64::max(f1,f2) > f64::max(f(a), f(b)) { return Err("No minima"); }

        for _ in 1..(((xtol/(b-a).abs()).log2()/GR.log2()) as i64) {
            if f1<f2 { b = x2; x2 = x1; f2 = f1; x1 = b - GR*(b-a); f1=f(x1); } 
            else { a = x1; x1 = x2; f1 = f2; x2 = a + GR*(b-a); f2 = f(x2); }
        }
        if (la-a).abs()<xtol || (lb-b).abs()<xtol { Err("No minima") } 
        else { Ok(((x1+x2)/2.0, (f1+f2)/2.0)) }
    }

    pub fn conj_grad_pr(
            f:impl Fn(&Array1<f64>)->f64, a:&Array1<f64>, delta:f64, 
            xtol:f64, g2tol:f64
        )->Result<Array1<f64>, &'static str> {

        fn gradient(f:&impl Fn(&Array1<f64>)->f64, x0:&Array1<f64>, 
                delta:f64, mut mult:f64)->Array1<f64> {
            mult /= delta; 
            let (mut x, f1, mut f2) = (x0.clone(), f(&x0), 0.0);
            x0.iter().zip(0..).map(|(&y,i)| { 
                x[i]=y+delta; f2 = (f(&x)-f1)*mult; x[i]=y; f2 
            } ).collect()
        }

        fn g_search(f:&impl Fn(&Array1<f64>)->f64, a:&Array1<f64>, z:&Array1<f64>, delta:f64, xtol:f64)->Result<(Array1<f64>, f64), &'static str> {

            fn b_phase(f:&impl Fn(&Array1<f64>)->f64, a0:&Array1<f64>, z0:&Array1<f64>, mut d:f64)->Result<(f64, f64), &'static str> {

                let (mut a, mut m, mut b) = (0.0, d, 2.0*d);
                let (mut fm, mut fb) = (
                    f(&(a0+&z0.mapv(|x| x*d))), f(&(a0+&z0.mapv(|x| x*b)))  
                );

                if fm > f(&a0) { return Err("No minima b-phase"); }

                for _ in 0..50 {
                    if fb > fm { return Ok((a,b)); }
                    else { 
                        d *= 2.0; a = m; m = b; fm = fb;  
                        b += d; fb = f(&(a0+&z0.mapv(|x| x*b)));
                    }
                }
                Err("No minima")
            }

            fn lin_search(fx:&impl Fn(&Array1<f64>)->f64, 
                av:&Array1<f64>, bv:&Array1<f64>, la:f64, lb:f64, xtol:f64)
                -> Result<(Array1<f64>, f64), &'static str> {

                const GR:f64 = 
                    0.618033988749894848204586834365638117720309179805762862135;

                let f = |x:f64| fx(&(av+&bv.mapv(|y| y*x))) ; 

                let (mut a, mut b) = (la, lb);
                let (mut x1, mut x2) = (b - GR*(b-a), a + GR*(b-a));
                let (mut f1, mut f2) = ( f(x1), f(x2) );

                if f64::max(f1,f2) > f64::max(f(a), f(b)) { return Err("No minima"); }

                for _ in 1..(((xtol/(b-a).abs()).log2()/GR.log2()) as i64) {
                    if f1<f2 { b=x2; x2=x1; f2=f1; x1=b-GR*(b-a); f1=f(x1); } 
                    else { a=x1; x1=x2; f1=f2; x2=a+GR*(b-a); f2=f(x2); }
                }
                Ok((av+&bv.mapv(|y| y*(x1+x2)/2.0), (f1+f2)/2.0))
            } 

            let (la,lb) = b_phase(&f, a, z, delta)?;
            lin_search(&f, a, z, la, lb, xtol)
        }    

        let n = a.len();
        let (mut x, mut g0, mut g1, mut d1) = 
            (a.clone(), a.clone(), a.clone(), a.clone());

        let mut g_sq;

        for _ in 0..30 {
            g0 = gradient(&f, &x, delta, 1.0);
            d1 = g0.mapv(|y| -y);
            for _ in 0..n {
                g_sq = g0.dot(&g0);
                if g_sq < g2tol { return Ok(x); }
                x = g_search(&f, &x, &d1, 0.005, xtol)?.0;
                g1 = gradient(&f, &x, delta, 1.0);
                d1 = d1.mapv(|y| y*((&g1-&g0).dot(&g1)/g_sq)) - &g1;
                g0 = g1;
            }
        }

        Err("No minima")
    }

}


pub mod roots {

    pub fn root_nwt(f1:impl Fn(f64)->f64, mut x:f64, xtol:f64)
            ->Result<f64, &'static str> { 
        let mut dx:f64; let del_x = xtol/10.0;
        for _ in 1..100 {
            dx = f1(x); dx = del_x * dx/ (f1(x+del_x) - dx); x -= dx;
            if dx.abs()<xtol { return Ok(x) }
        }
        Err("No soln")
    }

    pub fn root_nwt_der(f1:impl Fn(f64)->f64, fd:impl Fn(f64)->f64, mut x:f64, xtol:f64)->Result<f64, &'static str> { 
        let mut dx:f64;
        for _ in 1..100 {
            dx = f1(x)/fd(x); x -= dx; if dx.abs()<xtol { return Ok(x) }
        }
        Err("No soln")
    }


}


#[cfg(test)] mod basic {
    use super::*;

    #[test] fn app(){
        assert!(approx(1.00, 1.003, 1e-5) == false);
        assert!(approx(1.00, 1.03, 1e-2)  == false);
        assert!(approx(1.00, 1.003, 1e-2) == true);
        assert!(approx(1.00, 0.997, 1e-5) == false);
        assert!(approx(1.00, 0.99997, 1e-3) == true);
        assert!(approx(1.00, 1.003, 1e-5) == false);
        assert!(approx(1.00, 1.003, 1e-5) == false);
    }

    #[test] fn app_vec_test(){
        assert!(appr_vec(
            &vec![1.250005, 2.51, 1.46],
            &vec![1.25, 2.51, 1.46], 
            1e-5) == true);
        assert!(appr_vec(
            &vec![1.250005, 2.509995, 1.46],
            &vec![1.25, 2.51, 1.46], 
            1e-5) == true);
        assert!(appr_vec(
            &vec![1.25005, 2.51, 1.46],
            &vec![1.25, 2.51, 1.46], 
            1e-5) == false);
        assert!(appr_vec(
            &vec![1.25, 2.510005, 1.46001],
            &vec![1.25, 2.51, 1.46], 
            1e-5) == false);
    }

    #[test] fn app_arr_test(){
        assert!(approx_arr(
            array![1.250005, 2.51, 1.46].view(),
            array![1.25, 2.51, 1.46].view(), 
            1e-5) == true);
        assert!(approx_arr(
            array![1.250005, 2.509995, 1.46].view(),
            array![1.25, 2.51, 1.46].view(), 
            1e-5) == true);
        assert!(approx_arr(
            array![1.25005, 2.51, 1.46].view(),
            array![1.25, 2.51, 1.46].view(), 
            1e-5) == false);
        assert!(approx_arr(
            array![1.25, 2.510005, 1.46001].view(),
            array![1.25, 2.51, 1.46].view(), 
            1e-5) == false);
    }

    #[test] fn g_search_test(){
        // use super::*;
        assert!(approx(
            optim::g_search(|x| (x - 4.5).powf(2.0) + 5.0, 5.0, 1.0, 1e-10).unwrap().0, 4.5, 1e-7 ));
        assert!(approx(
            optim::g_search(|x| (x - 4.5).powf(2.0) + 5.0, 1.0, 5.0, 1e-10).unwrap().0, 4.5, 1e-7 ));
        assert!(optim::g_search(|x| -(x - 4.5).powf(2.0) + 5.0, 1.0, 5.0, 1e-10)     == Err("No minima") );
        assert!(optim::g_search(|x| -(x - 4.5).powf(2.0) + 5.0, 5.0, -1.0, 1e-10) == Err("No minima") );
    }

    #[test] fn conj_grad_test() {
        assert!(approx_arr(optim::conj_grad_pr(|x| (x[0]-3.0).powf(4.0) 
                    + (x[1]-4.0).powf(2.0) + (x[2]-2.0).powf(2.0) 
                    + (x[2]-2.0).powf(4.0) + 10.0,
                    &array![4.2,2.0,0.75], 1e-6, 1e-10, 1e-8).unwrap().view(), 
            array![3.0,4.0,2.0].view(), 1e-2));
    
        assert!(approx_arr(optim::conj_grad_pr(|x| (x[0]-3.0).powf(4.0) 
                    + (x[1]-4.0).powf(2.0) * (x[2]-2.0).powf(2.0) 
                    + (x[2]-2.0).powf(4.0) + (x[1]-4.0).powf(2.0) + 10.0,
                    &array![3.2,4.2,1.8], 1e-6, 1e-10, 1e-10).unwrap().view(), 
            array![3.0,4.0,2.0].view(), 1e-2));
    }

}


#[cfg(test)] mod roots_test {
    use super::*;

    #[test] fn root_nwt_der() {
        assert!(roots::root_nwt_der(|x| (x-4.0)*(x-3.0), 
            |x| (2.0*x-7.0), 2.0, 1e-6) == Ok(3.0));
        assert!(roots::root_nwt_der(|x| (x-4.0).powf(2.0), 
            |x| (2.0*x-8.0), 1.0, 1e-10) == Ok(3.99999999991268850863));
        assert!(roots::root_nwt_der(|x| (x-4.0).powf(2.0)+5.0, 
            |x| (2.0*x-8.0), 1.0, 1e-10) == Err("No soln"));
    }

    #[test] fn root_nwt() {
        assert!(roots::root_nwt(|x| (x-4.0)*(x-3.0), 
             2.0, 1e-6) == Ok(3.0));
        assert!(roots::root_nwt(|x| (x-4.0).powf(2.0), 
             1.0, 1e-6) == Ok(3.99999933598450496675));
        assert!(roots::root_nwt(|x| (x-4.0).powf(2.0)+5.0, 
             1.0, 1e-10) == Err("No soln"));
    }

}