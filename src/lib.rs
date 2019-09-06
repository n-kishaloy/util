

pub fn approx(x:f64, y:f64, xtol:f64)->bool { (x-y).abs() < xtol }

pub fn g_search(f1:impl Fn(f64)->f64,mut a0:f64,mut b0:f64,xtol:f64)->f64 {
    0.0
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