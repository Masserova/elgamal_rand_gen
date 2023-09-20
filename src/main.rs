#![allow(unused_imports)]
#![allow(warnings, unused)]

mod polynomials;
mod randomness_generator;
use randomness_generator::{RandomnessGenerator, PubParams};

fn main() {
    let t: u64 = 3;

    let pp = PubParams {t: t, n: 3*t + 1, n_parties_total: 5*t+4};
    let execution_leaks = false;
    //let vss: VSS<Fq> = VSS { secret: 1.into(), pp: pp, execution_leaks: execution_leaks};
    //vss.execute();

    let rand_generator = RandomnessGenerator { pp: pp, execution_leaks: execution_leaks}; 
    rand_generator.execute(); 

    
}