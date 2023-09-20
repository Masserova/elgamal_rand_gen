use std::{collections::BTreeMap};
use ark_ff::BigInteger;
use ark_std::test_rng;

use crate::polynomials::Poly;
//use ark_bls12_381::{Fq, Fr};
use curve25519_dalek::constants::{RISTRETTO_BASEPOINT_POINT};
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::traits::MultiscalarMul;
use std::mem::size_of_val;
use std::time::{SystemTime, Duration};
use rand::rngs::OsRng;
use std::ops::Mul;
use ark_std::rand::prelude::StdRng;
use ark_std::{One, Zero, UniformRand};
use ark_std::rand::SeedableRng;

pub struct PubParams {
    //Number of potentially adversarial parties
    pub t: u64,
    //Number of receivers
    pub n: u64,
    //Total umber of parties
    pub n_parties_total: u64
}

pub struct RandomnessGenerator {
    pub pp: PubParams,
    pub execution_leaks: bool
}


pub struct Dealer<'a> {
    pub pp: &'a PubParams,
    pub secret: Fq,
    pub g: Option<Fq>,
    pub h: Option<Fq>
}

pub struct Receiver<'a> {
    pub id: u64,
    pub pp: &'a PubParams
}


pub struct Reconstructor<'a> {
    pub pp: &'a PubParams
}

pub struct Client<'a> {
    pub pp: &'a PubParams
}

#[derive(Clone)]
pub struct Subshare {
    r: Fq,
    s: Fq
}

pub struct GeneratorPair {
    g: Fq,
    h: Fq
}


impl RandomnessGenerator {
    pub fn execute(&self) {
        let t = self.pp.t;
        let n = self.pp.n;

        let mut dealer_time = Duration::new(0,0);
        let mut receiver_time: Vec<Duration> = Vec::new();
        let mut reconstructor_time = Duration::new(0,0);
        let mut client_time = Duration::new(0,0);


        let mut dealer_comm =0.0;
        let mut receiver_comm: Vec<f64> = Vec::new();
        let mut reconstructor_comm = 0.0;
        let mut client_comm = 0.0;

        let mut dealer: Dealer = Dealer{secret: 1.into(), pp: &self.pp, g: None, h: None };

        let dealer_start_time = SystemTime::now();        
        //Dealer shares the secret, gather secret shares
        let (g, h, c, shares) = dealer.share();
        let dealer_end_time = SystemTime::now();
        dealer_time = dealer_end_time.duration_since(dealer_start_time).unwrap();
        println!("Dealer's work takes {} milliseconds", dealer_time.as_millis());

        //dealer_comm = (2.0*((size_of_val(&shares[1].1) as u64)*(shares.len() as u64) + (size_of_val(&c[1].1) as u64)*(c.len() as u64)  + (size_of_val(&g) as u64)) as f64)/1000000.0;


        //Each receiver verifies what it got from the dealer and what it got from other parties, compute what it wants to send to other parties
        for i in 1..=n {
            //need to forward these triply shares to the reconstructors
            let mut receiver_i: Receiver = Receiver{ id: i, pp: &self.pp };
            //need to forward these doubly shares to future receivers

            let receiver_start_time = SystemTime::now();
            let complain = receiver_i.receive_from_dealer(g, h, shares[i as usize].0, shares[i as usize].1, &c );
            let receiver_end_time = SystemTime::now();

            receiver_time.push(receiver_end_time.duration_since(receiver_start_time).unwrap());

            //receiver_comm.push(size_of_val(&complain) as u64 + 2.0*(size_of_val(&shares[&0].0) as u64)/1000000.0);
        }

        //Reconstructors publish projections that they received
        for _i in 1..=t+1 {
            let reconstructor: Reconstructor = Reconstructor{pp: &self.pp };
            for _j in 1..=n {
                reconstructor.receive_from_party(_j, shares[_j as usize].0, shares[_j as usize].1);
            }
        }
        //reconstructor_comm = (((size_of_val(&shares[&0].0) as f64)*2.0) as f64)/1000000.0;

        let client: Client = Client { pp: &self.pp };

        let client_start_time = SystemTime::now();
        let secret: (bool, Fq) = client.compute_secret(g, h, &c, &shares);
        let client_end_time = SystemTime::now();
        client_time = client_end_time.duration_since(client_start_time).unwrap();
        println!("Dealer's work takes {} milliseconds", dealer_time.as_millis());
        println!("First receiver's work takes {} milliseconds", receiver_time[0].as_millis());
        println!("Last receiver's work takes {} milliseconds", receiver_time[(n - 1) as usize].as_millis());
        println!("Client's work takes {} milliseconds", client_time.as_millis());


        println!("Dealer requires {} MB", dealer_comm);
        println!("First receiver requires {} MB", receiver_comm[0]);
        println!("Last receiver requires {} MB", receiver_comm[receiver_comm.len() - 1]);
        println!("Reconstructor requires {} MB", reconstructor_comm);


        let mut time_per_party: Vec<Duration> = Vec::new();
        let mut comm_per_party: Vec<f64> = Vec::new();

        let mut overall_time = 0;
        let mut overall_comm = 0.0;

        for i in 1..=5*t + 4 {
            time_per_party.push(Duration::new(0,0));
            comm_per_party.push(0.0);
        }

        //first t+1 parties are acting as dealers
        for i in 1..=t + 1 {
            time_per_party[(i - 1) as usize] += dealer_time;
            comm_per_party[(i - 1) as usize] += dealer_comm;

            for k in (i + 1)..=3*t + i + 1 {
                time_per_party[(k - 1) as usize] += receiver_time[(k - 1 - i) as usize];
                comm_per_party[(k - 1) as usize] += receiver_comm[(k - 1 - i) as usize];
            }
        }

        for k in 4*t + 4..=5*t + 4 {
            comm_per_party[(k - 1) as usize] += reconstructor_comm;
        }

        for i in 1..=5*t + 4 {
            overall_time += time_per_party[(i-1) as usize].as_millis();
            overall_comm += comm_per_party[(i-1) as usize];
        }

        overall_time += client_time.as_millis();

        println!("Overall time: {}", overall_time);
        println!("Overall comm: {}", overall_comm);


        dbg!(secret);

    }
}

impl Dealer<'_> {
    fn set_generator_pair(&mut self) {
        let mut csprng = OsRng{};
        let a = Fq::rand(&mut csprng);
        let G = Fq::rand(&mut csprng);
        let H = G.mul(a);
        self.g = Some(G);
        self.h = Some(H);
    }

    pub fn share(&mut self) -> (Fq, Fq, Vec<(Fq, Fq)>, Vec<(Fq, Fq)>) {
        let n = self.pp.n;

        let mut csprng: StdRng = StdRng::seed_from_u64(6);
        // Generating two random polynomials f1 and f2
        let f1 = Poly::<Fq>::rand(self.pp.t, &mut csprng);
        let mut f2 = Poly::<Fq>::rand(self.pp.t, &mut csprng);

        //Set f2(0) = s
        f2.coeffs[0] = self.secret;

        // Generate a pair of generators g and h
        self.set_generator_pair();

        let g = self.g.unwrap();        
        let h = self.h.unwrap();

        let c: Vec<(Fq, Fq)> = (0..=self.pp.t)
            .map (|i| (g.mul(f1.coeffs[i as usize]), g.mul(f1.coeffs[i as usize]) + h.mul(f2.coeffs[i as usize])))
            .collect();

        let shares: Vec<(Fq, Fq)> = (1..=n)
            .map(|i| (f1.eval(i.into()), f2.eval(i.into())))
            .collect();

        (g, h, c, shares)
    }
}


impl Receiver<'_> {


    pub fn receive_from_dealer(&mut self, g: Fq, h: Fq, r: Fq, s: Fq, c: &Vec<(Fq, Fq)>) 
                                                    -> bool {
        let mut complain = false;

        let mut left = Fq::from(0);
        let mut powi = 1;

        for k in (0..=self.pp.t) {
            left += c[k as usize].0.mul(powi.into());
            powi *= self.id;
        }

        let mut right = Fq::from(0);
        powi = 1;

        for k in (0..=self.pp.t) {
            right += c[k as usize].1.mul(powi.into());
            powi *= self.id;
        }

        if (r != left || s != right) {
            complain = true;
        }
        complain
    }
}

impl Reconstructor<'_> {
    pub fn receive_from_party(&mut self, id: u64, r: Fq, s: Fq) -> (Fq, Fq) {
        //check if anyone broadcast complains, otherwise just output the share

        (r, s)
    }
}


impl Client<'_> {
    pub fn compute_secret(&self, g: Fq, h: Fq, c: &Vec<(Fq, Fq)>, shares: &Vec<(Fq, Fq)>) -> (bool,Fq) {
        let mut secret_computable = true; 
        let n = self.pp.n;
        let t = self.pp.t;

        let mut verified_shares_keys: Vec<u64> = Default::default();
        let mut verified_shares_values: Vec<Fq> = Default::default();
        let mut n_verified_poly = 0;

        for i in 1..=n {
            if (g.mul(shares[i as usize].1) != c[i as usize].0) && (g.mul(shares[i as usize].0) + h.mul(shares[i as usize].1) == c[i as usize].1) {
                verified_shares_keys.push(i);
                verified_shares_values.push(shares[i as usize].1);
            }
        }


        if n_verified_poly < t  + 1 {
            println!("I'm unhappy!")
        }

        let unipoly: Poly<Fq> = Poly::evals_to_coeffs(&verified_shares_keys, &verified_shares_values, n_verified_poly);

        let secret: Fq = unipoly.eval(Fq::from(0));
        secret_computable = n_verified_poly > t;
        (secret_computable, secret)
    }
}
