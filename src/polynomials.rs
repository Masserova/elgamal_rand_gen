use std::{collections::BTreeMap};

use ark_ff::{Field};
use ark_std::rand::Rng;
/// Symmetric bi-variate polynomial
#[derive(Debug)]

pub struct Poly<F: Field> {
    pub coeffs: Vec<F>,
    pub degree: u64,
}

impl<F: Field> Poly<F> {
    pub fn eval(&self, x: F) -> F {
        let mut result = self.coeffs[self.degree as usize];
        for deg_x in (1..=self.degree).rev() {
            result = self.coeffs[(deg_x-1) as usize] + x*result;
        }
        result
    }
    pub fn rand<R: Rng>(d: u64, rng: &mut R) -> Poly<F> {
        Poly {
            coeffs: [0..d]
                .map(|deg| F::rand(rng))
                .to_vec(),
            degree: d,
        }
    }

    pub fn evals_to_coeffs(x: &Vec<u64>, y: &Vec<F>, n: u64) -> Poly<F> {
        let mut full_coeffs: Vec<F> = vec![F::ZERO; n as usize];
        let mut terms: Vec<F> = vec![F::ZERO; n as usize];

        let mut prod: F;
        let mut degree = 0;
        for i in 0..=n-1 {
            prod = F::ONE; 

            for _j in 0..=n-1 {
                terms[_j as usize] = F::ZERO;
            }

            for j in 0..=n-1 {
                if i == j {
                    continue;
                } 
                prod *= F::from(x[i as usize]) - F::from(x[j as usize]);
            }

            prod = y[i as usize] / prod;

            terms[0] = prod;

            for j in 0..=n-1 {
                if i == j {
                    continue;
                }
                for k in (1..n).rev() {
                    let tmp_term = terms[(k - 1) as usize];
                    //dbg!(k, tmp_term);
                    terms[k as usize] += tmp_term;
                    terms[(k - 1) as usize] *= -F::from(x[j as usize]);
                }
            }

            for j in 0..=n-1 {
                full_coeffs[j as usize] += terms[j as usize];
            }
        }

        for j in (0..=n-1).rev() {
            if full_coeffs[j as usize] != F::ZERO {
                //dbg!(j);
                degree = j;
                break;
            }
        }

        Poly {
            degree: degree,
            coeffs: full_coeffs
        }

    }
}
