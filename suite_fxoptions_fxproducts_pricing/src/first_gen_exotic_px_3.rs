use rand::prelude::*;
use rand_distr::{Normal, Distribution};
use statrs::distribution::{ContinuousCDF, Normal as StatNormal};

struct ModelParams {
    s0: f64,      // Initial exchange rate
    mu: f64,      // Drift (for GBM/BM with drift)
    sigma: f64,   // Volatility
    t: f64,       // Time to maturity
}

/// Case: One-Touch Digital Hedge via Static Replication (Approx)
/// Pays 1 if barrier B is hit.
fn one_touch_price(p: &ModelParams, b: f64, steps: usize, sims: usize) -> f64 {
    let dt = p.t / steps as f64;
    let mut rng = thread_rng();
    let mut hits = 0;

    for _ in 0..sims {
        let mut curr_s = p.s0;
        let mut touched = false;
        for _ in 0..steps {
            let dw = Normal::new(0.0, dt.sqrt()).unwrap().sample(&mut rng);
            // GBM: dS = mu*S*dt + sigma*S*dW
            curr_s *= ( (p.mu - 0.5 * p.sigma.powi(2)) * dt + p.sigma * dw ).exp();
            
            if (b > p.s0 && curr_s >= b) || (b < p.s0 && curr_s <= b) {
                touched = true;
                break;
            }
        }
        if touched { hits += 1; }
    }
    hits as f64 / sims as f64
}

/// Case B: Double One-Touch (DOT)
/// Pays 1 if either Upper Barrier (U) or Lower Barrier (L) is hit.
fn double_one_touch_price(p: &ModelParams, l: f64, u: f64, steps: usize, sims: usize) -> f64 {
    let dt = p.t / steps as f64;
    let mut rng = thread_rng();
    let mut hits = 0;

    for _ in 0..sims {
        let mut curr_s = p.s0;
        let mut touched = false;
        for _ in 0..steps {
            let dw = Normal::new(0.0, dt.sqrt()).unwrap().sample(&mut rng);
            curr_s *= ( (p.mu - 0.5 * p.sigma.powi(2)) * dt + p.sigma * dw ).exp();
            
            if curr_s >= u || curr_s <= l {
                touched = true;
                break;
            }
        }
        if touched { hits += 1; }
    }
    hits as f64 / sims as f64
}

pub fn main() {
    // Synthetic eg of EURSGD one touch option 
    // mu = interest rate differential (SGD rate ~ 0.9% and ECB rate ~ 2.15%)
    let params = ModelParams { s0: 1.4951, mu: 0.012, sigma: 0.065, t: 0.25 };
    
    // 1. One-Touch with constant Vol (GBM) - upper barrier where option would pay out if EURSGD
    // touches 1.52 at any point in the next 3 months
    let ot_price = one_touch_price(&params, 1.52, 252, 100_000);
    println!("One-Touch (B=1.2) Price: {:.4}", ot_price);

    // 2. Double One-Touch (L=1.45, U=1.52)
    let dot_price = double_one_touch_price(&params, 1.45, 1.52, 252, 100_000);
    println!("Double One-Touch (L=0.8, U=1.2) Price: {:.4}", dot_price);
}

// Scenario 1 - No drift: for one-touch, price depends purely on distance to barrier and volatility, 
// symmetric for up/down barriers. For double one-touch: price is higher than single one-touch but lower
// than sum of 2 separate one-touches due to path overlap.
// Scenario 2 - Positive drift: For one-touch, upward one-touch price increases, downward one-touch price decreases significantly
// For double one-touch, probability shifts towards upper barrier, sensitivity to lower barrier vanishes as \meu * T grows
// Scenario 3 - Higher volatility: Probability of hitting barrier increases for ITM and OTM barriers. For double one-touch, 
// DOT price -> 1 as even narrow corridors become likely to be breached.
// For hedging, drift makes static replication harder because then need to adjust weight of replicating portfolio by
// (B/S_{0})^{2*\meu / \sigma^{2}}