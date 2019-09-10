

pub fn approx(x:f64, y:f64, xtol:f64)->bool { (x-y).abs() < xtol }

pub fn g_search(f:impl Fn(f64)->f64,mut a:f64,mut b:f64,xtol:f64)
        ->Result<(f64,f64), &'static str> {

    const GR:f64= 0.6180339887498948482045868343656381177203091798057628621354;
    let mut f1  = f64::min(a,b); b = f64::max(a,b); a = f1;

    let (mut x1, mut x2) = (b - GR*(b-a), a + GR*(b-a));
    f1 = f(x1); let mut f2 = f(x2);

    for _ in 1..(((xtol/(b-a)).log2()/GR.log2()) as i64) {
        if f1 < f2 { b = x2; x2 = x1; f2 = f1; x1 = b - GR*(b-a); f1 = f(x1); } 
        else { a = x1; x1 = x2; f1 = f2; x2 = a + GR*(b-a); f2 = f(x2); }
    }
    Ok(((x1+x2)/2.0, (f1+f2)/2.0))
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

    #[test] fn g_search_test(){
        assert!(g_search(|x| (x - 4.5).powf(2.0) + 5.0, 1.0, 5.0, 1e-15) 
        == Ok((4.5000000210734238947907215, 5.0)) );
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