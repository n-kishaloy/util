

pub fn approx(x:f64, y:f64, xtol:f64)->bool { (x-y).abs() < xtol }

pub fn appr_vec(a:&Vec<f64>, b:&Vec<f64>, ztol:f64)->bool {
    a.iter().zip(b).fold(0.0,|s,(x,y)| s+(x-y).powf(2.0)) < ztol.powf(2.0)
}

pub fn addvec(a:&Vec<f64>, b:&Vec<f64>)->Vec<f64> { 
    a.iter().zip(b).map(|(x,y)|x+y).collect()
}

pub fn subvec(a:&Vec<f64>, b:&Vec<f64>)->Vec<f64> { 
    a.iter().zip(b).map(|(x,y)|x-y).collect()
}

pub fn mulvec(a:&Vec<f64>, b:f64)->Vec<f64> { a.iter().map(|x| x*b).collect() }

pub fn dotvec(a:&Vec<f64>, b:&Vec<f64>)->f64 { 
    a.iter().zip(b).fold(0.0,|s,(x,y)| s+x*y) 
}

pub fn explvec(a:&Vec<f64>, b:&Vec<f64>, d:f64)->Vec<f64> {
    a.iter().zip(b).map(|(x,y)|x + d*y).collect()
}

pub fn mxplvec(a:&Vec<f64>, b:&Vec<f64>, da:f64, db:f64)->Vec<f64> {
    a.iter().zip(b).map(|(x,y)|da*x + db*y).collect()
}

pub mod optim {
        
    use crate::{addvec, mulvec, explvec, mxplvec, subvec, dotvec};
    
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

    pub fn conj_grad_PR(
            f:impl Fn(&Vec<f64>)->f64, a:&Vec<f64>, delta:f64, 
            xtol:f64, g2tol:f64
        )->Result<Vec<f64>, &'static str> {

        fn gradient(f:&impl Fn(&Vec<f64>)->f64, x0:&Vec<f64>, 
                delta:f64, mut mult:f64)->Vec<f64> {
            mult /= delta; 
            let (mut x, f1, mut f2) = (x0.clone(), f(&x0), 0.0);
            x0.iter().zip(0..).map(|(&y,i)| { 
                x[i]=y+delta; f2 = (f(&x)-f1)*mult; x[i]=y; f2 
            } ).collect()
        }

        fn g_search(f:&impl Fn(&Vec<f64>)->f64, a:&Vec<f64>, z:&Vec<f64>, delta:f64, xtol:f64)->Result<(Vec<f64>, f64), &'static str> {

            fn b_phase(f:&impl Fn(&Vec<f64>)->f64, a0:&Vec<f64>, z0:&Vec<f64>, mut d:f64)->Result<(f64, f64), &'static str> {

                let (mut a, mut m, mut b) = (0.0, d, 2.0*d);
                let (mut fm, mut fb) = (
                    f(&explvec(&a0,&z0,d)), f(&explvec(&a0,&z0,2.0*d))  
                );

                if fm > f(&a0) { return Err("No minima b-phase"); }

                for _ in 0..50 {
                    if fb > fm { return Ok((a,b)); }
                    else { 
                        d *= 2.0; a = m; m = b; fm = fb;  
                        b += d; fb = f(&explvec(&a0, &z0, b));
                    }
                }
                Err("No minima")
            }

            fn lin_search(fx:&impl Fn(&Vec<f64>)->f64, 
                av:&Vec<f64>, bv:&Vec<f64>, la:f64, lb:f64, xtol:f64)
                -> Result<(Vec<f64>, f64), &'static str> {

                const GR:f64 = 
                    0.618033988749894848204586834365638117720309179805762862135;

                let f = |x:f64| fx(&explvec(&av, &bv, x)) ; 

                let (mut a, mut b) = (la, lb);
                let (mut x1, mut x2) = (b - GR*(b-a), a + GR*(b-a));
                let (mut f1, mut f2) = ( f(x1), f(x2) );

                if f64::max(f1,f2) > f64::max(f(a), f(b)) { return Err("No minima"); }

                for _ in 1..(((xtol/(b-a).abs()).log2()/GR.log2()) as i64) {
                    if f1<f2 { b=x2; x2=x1; f2=f1; x1=b-GR*(b-a); f1=f(x1); } 
                    else { a=x1; x1=x2; f1=f2; x2=a+GR*(b-a); f2=f(x2); }
                }
                Ok((explvec(&av,&bv,(x1+x2)/2.0), (f1+f2)/2.0)) 
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
            d1 = mulvec(&g0, -1.0);
            for _ in 0..n {
                g_sq = dotvec(&g0, &g0);
                if g_sq < g2tol { return Ok(x); }
                x = g_search(&f, &x, &d1, 0.005, xtol)?.0;
                g1 = gradient(&f, &x, delta, 1.0);
                d1 = mxplvec(&g1,&d1,-1.0, dotvec(&subvec(&g1,&g0),&g1)/g_sq);
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

    #[test] fn g_search_test(){
        // use super::*;
        assert!(approx(
            optim::g_search(|x| (x - 4.5).powf(2.0) + 5.0, 5.0, 1.0, 1e-10).unwrap().0, 4.5, 1e-7 ));
        assert!(approx(
            optim::g_search(|x| (x - 4.5).powf(2.0) + 5.0, 1.0, 5.0, 1e-10).unwrap().0, 4.5, 1e-7 ));
        assert!(optim::g_search(|x| -(x - 4.5).powf(2.0) + 5.0, 1.0, 5.0, 1e-10)     == Err("No minima") );
        assert!(optim::g_search(|x| -(x - 4.5).powf(2.0) + 5.0, 5.0, -1.0, 1e-10) == Err("No minima") );
    }

    #[test] fn vec_tests() {
        assert!(dotvec(&vec![1.2, 2.3, -1.5], &vec![2.1, -1.5, 3.5]) == -6.18);
        assert!(appr_vec(&addvec(&vec![1.2, 2.3,-2.8], &vec![2.5,-5.2, 1.2] ), 
            &vec![3.7,-2.9,-1.6], 1e-8));
        assert!(appr_vec(&subvec(&vec![1.2, 2.3,-2.8], &vec![2.5,-5.2, 1.2] ), 
            &vec![-1.3,7.5,-4.0], 1e-8));
        assert!(appr_vec(&mulvec(&vec![1.2, 2.4, -3.5], 2.0), 
            &vec![2.4, 4.8,-7.0], 1e-8));
        assert!(appr_vec(&explvec(&vec![2.4, 1.5, -3.0], &vec![1.8, 1.0, 2.5], 2.0), &vec![6.0, 3.5, 2.0], 1e-8));
        assert!(appr_vec(&mxplvec(&vec![2.4, 1.5, -3.0], &vec![1.8, 1.0, 2.5], 2.0, 3.0), &vec![10.2, 6.0, 1.5], 1e-8));
    }

    #[test] fn conj_grad_test() {
        assert!(appr_vec(&optim::conj_grad_PR(|x| (x[0]-3.0).powf(4.0) 
                    + (x[1]-4.0).powf(2.0) + (x[2]-2.0).powf(2.0) 
                    + (x[2]-2.0).powf(4.0) + 10.0,
                    &vec![4.2,2.0,0.75], 1e-6, 1e-10, 1e-8).unwrap(), 
            &vec![3.0,4.0,2.0], 1e-2));
    
        assert!(appr_vec(&optim::conj_grad_PR(|x| (x[0]-3.0).powf(4.0) 
                    + (x[1]-4.0).powf(2.0) * (x[2]-2.0).powf(2.0) 
                    + (x[2]-2.0).powf(4.0) + (x[1]-4.0).powf(2.0) + 10.0,
                    &vec![3.2,4.2,1.8], 1e-6, 1e-10, 1e-10).unwrap(), 
            &vec![3.0,4.0,2.0], 1e-2));
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