use ark_ff::BigInt;
use curve25519_dalek::scalar::Scalar;
use rand_core::OsRng;

/// Symmetric bi-variate polynomial
#[derive(Debug)]

pub struct Poly {
    pub coeffs: Vec<Scalar>,
    pub degree: u64,
}

impl Poly {
    pub fn eval(&self, x: Scalar ) -> Scalar {
        let mut result = self.coeffs[self.degree as usize];
        for deg_x in (1..=self.degree).rev() {
            result = self.coeffs[(deg_x-1) as usize] + x*result;
        }
        result
    }
    pub fn rand(d: u64, rng: &mut OsRng) -> Poly {
        Poly {
            coeffs: (0..=d)
                .into_iter()
                .map(|deg| Scalar::random(rng))
                .collect(),                
            degree: d,
        }
    }

    pub fn evals_to_coeffs(x: &Vec<u64>, y: &Vec<Scalar>, n: u64) -> Poly {
        let mut full_coeffs: Vec<Scalar> = vec![Scalar::ZERO; n as usize];
        let mut terms: Vec<Scalar> = vec![Scalar::ZERO; n as usize];

        let mut prod: Scalar;
        let mut degree = 0;
        for i in 0..=n-1 {
            prod = Scalar::ONE; 

            for _j in 0..=n-1 {
                terms[_j as usize] = Scalar::ZERO;
            }

            for j in 0..=n-1 {
                if i == j {
                    continue;
                } 
                prod *= Scalar::from(x[i as usize])- Scalar::from(x[j as usize]);
            }

            prod = y[i as usize] * prod.invert();

            terms[0] = prod;

            for j in 0..=n-1 {
                if i == j {
                    continue;
                }
                for k in (1..n).rev() {
                    let tmp_term = terms[(k - 1) as usize];
                    //dbg!(k, tmp_term);
                    terms[k as usize] += tmp_term;
                    terms[(k - 1) as usize] *= -Scalar::from(x[j as usize]);
                }
            }

            for j in 0..=n-1 {
                full_coeffs[j as usize] += terms[j as usize];
            }
        }

        for j in (0..=n-1).rev() {
            if full_coeffs[j as usize] != Scalar::ZERO {
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
