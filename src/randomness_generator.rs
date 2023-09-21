use ark_ff::BigInteger;
use ark_std::test_rng;

use crate::polynomials::Poly;
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::traits::{MultiscalarMul, Identity};
use std::mem::size_of_val;
use std::time::{SystemTime, Duration};
use rand_core::OsRng;
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
    pub g: Option<RistrettoPoint>,
    pub h: Option<RistrettoPoint>
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
    r: Scalar,
    s: Scalar
}

pub struct GeneratorPair {
    g: RistrettoPoint,
    h: RistrettoPoint
}


impl RandomnessGenerator {
    pub fn execute(&self) {
        let t = self.pp.t;
        let n = self.pp.n;

        let mut dealer_time = Duration::new(0,0);
        let mut receiver_time: Vec<Duration> = Vec::new();
        let mut reconstructor_time = Duration::new(0,0);
        let mut client_time = Duration::new(0,0);



        let mut overall_time = 0;
        let mut overall_comm = 0.0;
        
        let mut dealer_comm =0.0;
        let mut receiver_comm: Vec<f64> = Vec::new();
        let mut reconstructor_comm = 0.0;
        let mut client_comm = 0.0;

        let mut f1: Poly;
        let mut f2: Poly;

        let mut dealers: Vec<Dealer> = Default::default();
        let mut receivers: Vec<Receiver> = Default::default();
        let mut receivers: Vec<Receiver> = Default::default();
        let mut reconstructors: Vec<Reconstructor> = Default::default();

        for i in 1..=t+1 {
            let dealer: Dealer = Dealer{pp: &self.pp, g: None, h: None };
            dealers.push(dealer);
        }

        for i in 1..=n {
            let receiver: Receiver = Receiver{ id: i, pp: &self.pp };
            let reconstructor: Reconstructor = Reconstructor{pp: &self.pp };
            receivers.push(receiver);
            reconstructors.push(reconstructor);
        }

        let dealer_start_time = SystemTime::now();        
        //Dealer shares the secret, gather secret shares
        let (g, h, c, shares, f1, f2) = dealers[0].share();
        let dealer_end_time = SystemTime::now();
        dealer_time = dealer_end_time.duration_since(dealer_start_time).unwrap();


        //println!("Dealer's work takes {} milliseconds", dealer_time.as_millis());

        dealer_comm = (2.0*(size_of_val(&f1.coeffs[0])*f1.coeffs.len() + size_of_val(&f1.degree) + size_of_val(&shares[1].1)*shares.len() + size_of_val(&c[1].1)*c.len() + size_of_val(&g)) as f64)/1000000.0;


        //Each receiver verifies what it got from the dealer and what it got from other parties, compute what it wants to send to other parties
        for i in 1..=n {
            //need to forward these triply shares to the reconstructors
            let mut receiver_i: Receiver = Receiver{ id: i, pp: &self.pp };
            //need to forward these doubly shares to future receivers

            let receiver_start_time = SystemTime::now();
            let complain = receiver_i.receive_from_dealer(g, h, shares[i as usize - 1].0, shares[i as usize - 1].1, &c );
            let receiver_end_time = SystemTime::now();

            receiver_time.push(receiver_end_time.duration_since(receiver_start_time).unwrap());

            receiver_comm.push(2.0*(size_of_val(&shares[0].0)) as f64/1000000.0);
        }

        //Reconstructors publish projections that they received
        for _i in 1..=t+1 {
            let mut reconstructor: Reconstructor = Reconstructor{pp: &self.pp };
            for _j in 1..=n {
                reconstructor.receive_from_party(_j, shares[_j as usize - 1].0, shares[_j as usize - 1].1);
            }
        }
        reconstructor_comm = (2.0*(size_of_val(&shares[0].0)) as f64/1000000.0);

        let client: Client = Client { pp: &self.pp };

        let client_start_time = SystemTime::now();
        let secret: (bool, Scalar) = client.compute_secret(g, h, &c, &shares);
        let client_end_time = SystemTime::now();
        client_time = client_end_time.duration_since(client_start_time).unwrap();
        println!("Dealer's work takes {} milliseconds", dealer_time.as_millis());
        println!("First receiver's work takes {} milliseconds", receiver_time[0].as_millis());
        println!("Client's work takes {} milliseconds", client_time.as_millis());


        println!("Dealer requires {} MB", dealer_comm);
        println!("First receiver requires {} MB", receiver_comm[0]);
        println!("Reconstructor requires {} MB", reconstructor_comm);


        let mut time_per_party: Vec<Duration> = Vec::new();
        let mut comm_per_party: Vec<f64> = Vec::new();

        for i in 1..=5*t + 4 {
            time_per_party.push(Duration::new(0,0));
            comm_per_party.push(0.0);
        }

        //first t+1 parties are acting as dealers
        for i in 1..=t + 1 {
            time_per_party[(i - 1) as usize] += dealer_time;
            comm_per_party[(i - 1) as usize] += dealer_comm;

            for k in (i + 1)..=2*t + i + 1 {
                time_per_party[(k - 1) as usize] += receiver_time[0 as usize];
                comm_per_party[(k - 1) as usize] += receiver_comm[0 as usize];
            }
        }

        for k in 3*t + 4..=5*t + 4 {
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
        let G = RistrettoPoint::random(&mut csprng);
        let H = RistrettoPoint::random(&mut csprng);
        self.g = Some(G);
        self.h = Some(H);
    }

    pub fn share(&mut self) -> (RistrettoPoint, RistrettoPoint, Vec<(RistrettoPoint, RistrettoPoint)>, Vec<(Scalar, Scalar)>, Poly, Poly) {
        let n = self.pp.n;

        let mut csprng = OsRng{};
        // Generating two random polynomials f1 and f2
        let f1 = Poly::rand(self.pp.t, &mut csprng);
        let mut f2 = Poly::rand(self.pp.t, &mut csprng);

        // Generate a pair of generators g and h
        self.set_generator_pair();

        let g = self.g.unwrap();        
        let h = self.h.unwrap();

        let c: Vec<(RistrettoPoint, RistrettoPoint)> = (0..=self.pp.t)
            .map (|i| (g.mul(f1.coeffs[i as usize]), RistrettoPoint::multiscalar_mul([f1.coeffs[i as usize], f2.coeffs[i as usize]], [g,h])))
            .collect();

        let shares: Vec<(Scalar, Scalar)> = (1..=n)
            .map(|i| (f1.eval(Scalar::from(i)), f2.eval(Scalar::from(i))))
            .collect();

        (g, h, c, shares, f1, f2)
    }
}


impl Receiver<'_> {


    pub fn receive_from_dealer(&mut self, g: RistrettoPoint, h: RistrettoPoint, r: Scalar, s: Scalar, c: &Vec<(RistrettoPoint, RistrettoPoint)>) 
                                                    -> bool {
        let mut complain = false;

        let powers: Vec<u64> =  (0..=self.pp.t)
                            .map(|i| u64::pow(self.id,i as u32))
                            .collect();

        let scalars: Vec<Scalar> = (0..=self.pp.t)
                            .map(|i| Scalar::from(powers[i as usize]))
                            .collect();

        let points: Vec<RistrettoPoint> = (0..=self.pp.t)
                                            .map(|i| c[i as usize].0)
                                            .collect();

        let left = RistrettoPoint::multiscalar_mul(&scalars, points);

        let points: Vec<RistrettoPoint> = (0..=self.pp.t)
                                            .map(|i| c[i as usize].1)
                                            .collect();

        let right = RistrettoPoint::multiscalar_mul(scalars, points);
        

        if g.mul(r) != left || RistrettoPoint::multiscalar_mul([r,s],[g,h]) != right {
            complain = true;
        }
        complain
    }
}

impl Reconstructor<'_> {
    pub fn receive_from_party(&mut self, id: u64, r: Scalar, s: Scalar) -> (Scalar, Scalar) {
        //check if anyone broadcast complains, otherwise just output the share

        (r, s)
    }
}


impl Client<'_> {
    pub fn compute_secret(&self, g: RistrettoPoint, h: RistrettoPoint, c: &Vec<(RistrettoPoint, RistrettoPoint)>, shares: &Vec<(Scalar, Scalar)>) -> (bool,Scalar) {
        let mut secret_computable = true; 
        let n = self.pp.n;
        let t = self.pp.t;

        let mut verified_shares_keys: Vec<u64> = Default::default();
        let mut verified_shares_values: Vec<Scalar> = Default::default();


        for i in 1..=n {
            let powers: Vec<u64> =  (0..=self.pp.t)
                            .map(|j| u64::checked_pow(i,j as u32).unwrap())
                            .collect();

            let scalars: Vec<Scalar> = (0..=self.pp.t)
                                .map(|j| Scalar::from(powers[j as usize]))
                                .collect();

            let points: Vec<RistrettoPoint> = (0..=self.pp.t)
                                                .map(|j| c[j as usize].0)
                                                .collect();

            let left = RistrettoPoint::multiscalar_mul(&scalars, points);

            let points: Vec<RistrettoPoint> = (0..=self.pp.t)
                                                .map(|j| c[j as usize].1)
                                                .collect();

            let right = RistrettoPoint::multiscalar_mul(scalars, points);
            let r = shares[i as usize - 1].0;
            let s = shares[i as usize - 1].1;

            if g.mul(r) == left && RistrettoPoint::multiscalar_mul([r,s],[g,h]) == right {
                verified_shares_keys.push(i);
                verified_shares_values.push(shares[i as usize - 1].1);
            }
        }

        let unipoly: Poly = Poly::evals_to_coeffs(&verified_shares_keys, &verified_shares_values, n);

        let secret: Scalar = unipoly.eval(Scalar::ZERO);
        secret_computable = unipoly.degree <= t;
        (secret_computable, secret)
    }
}
