use statrs::distribution::{ContinuousCDF, Normal};
use rand::prelude::*;
use rand_distr::{StandardNormal, Distribution};

fn price_perpetual_one_touch(s: f64, h: f64, rd: f64, rf: f64, sigma: f64) -> f64 {
    if (s >= h && s > 0.0) || (s <= h && s < 0.0) { return 1.0; }
    let b = rd - rf;
    let alpha = (-(b - 0.5 * sigma.powi(2)) + ((b - 0.5 * sigma.powi(2)).powi(2) + 2.0 * sigma.powi(2) * rd).sqrt()) / sigma.powi(2);
    (s / h).powf(alpha)
}

fn price_dop(s: f64, k: f64, h: f64, t: f64, rd: f64, rf: f64, sigma: f64) -> f64 {
    if s <= h { return 0.0; }
    let n = Normal::new(0.0, 1.0).unwrap();
    let b = rd - rf; // Cost of carry
    
    // Standard Vanilla Put (Black-Scholes)
    let d1 = ((s/k).ln() + (b + 0.5 * sigma.powi(2)) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    let vanilla_put = k * (-rd * t).exp() * n.cdf(-d2) - s * (-rf * t).exp() * n.cdf(-d1);
    
    // Barrier Adjustment (Down-and-In component to be subtracted)
    let mu = (b - 0.5 * sigma.powi(2)) / sigma.powi(2);
    let y = ( (h.powi(2) / (s * k)).ln() ) / (sigma * t.sqrt()) + (mu + 1.0) * sigma * t.sqrt();
    
    let di_put = s * (-rf * t).exp() * (h/s).powf(2.0 * (mu + 1.0)) * n.cdf(y) 
               - k * (-rd * t).exp() * (h/s).powf(2.0 * mu) * n.cdf(y - sigma * t.sqrt());

    (vanilla_put - di_put).max(0.0)
}
// in case of all-in = spot, if we increase `sigma` (i.e. volatility), `di_put` grows much faster
// than vanilla put. Since DOP = Vanilla - DI (down-and-in), total value of down-and-out put falls
// as volatility increases. Extension: vega analysis will be able to show peak of value before
// knock-out rise starts cancelling it out.

// Sensitivity analysis to spot and forward price changes
// Impact of spot price chg: near barrier, delta becomes positive (option val falls with spot) and
// gamma becomes very negative
pub fn main() {
    let k = 150.0;    // Strike
    let h = 140.0;     // Lower Barrier
    let t = 0.5;      // 6 Months
    let sigma = 0.20; // 20% Vol
    println!("Pricing down-and-out knock-out put option synthetic egs for USDJPY:");
    println!("{:<10} | {:<15} | {:<15}", "Spot", "Fwd > Spot", "Fwd < Spot");
    println!("{:-<45}", "");

    // Iterate through spot prices from 141 to 160
    for i in 0..=10 {
        let s = h + 1.0 + (i as f64 * 1.9);
        
        // Scenario 1: rd = 5%, rf = 2% (Forward trades at a premium to Spot)
        let price_fwd_up = price_dop(s, k, h, t, 0.05, 0.02, sigma);
        
        // Scenario 2: rd = 2%, rf = 5% (Forward trades at a discount to Spot)
        let price_fwd_down = price_dop(s, k, h, t, 0.02, 0.05, sigma);
        
        println!("{:<10.2} | {:<15.4} | {:<15.4}", s, price_fwd_up, price_fwd_down);
    }

    // code to show that if EURGBP XR negatively corr w GBP interest rates, how does this change TV of
    // strike-out put. Synthetic eg of EURGBP
    let s0 = 0.85;       // Current EURGBP Spot
    let k_tv = 0.86;     // Strike Price
    let h_tv = 0.82;     // Down-and-Out Barrier
    let T = 1.0;         // Time to Expiry (1 Year)
    let rf = 0.02;       // Foreign Rate (EUR)
    let rd_base = 0.04;  // Base Domestic Rate (GBP)
    let sigma_s = 0.10;  // FX Volatility
    let sigma_r = 0.01;  // Rate Volatility
    let paths: usize = 100_000;
    main_xr_rate_tv(s0, k_tv, h_tv, T, rf, rd_base, sigma_s, sigma_r, paths);
}


// code uses Cholesky decomposition to model FX with interest rates.
// `knocked_out` flag ensures that if spot touches `h`, payoff is zero
// negative `rho` reinforces "upward" drift of forward price when spot falling, making put less valuable
// as it approaches strike
fn main_xr_rate_tv(s0: f64, k: f64, h: f64, T: f64, rf: f64, rd_base: f64, sigma_s: f64, sigma_r: f64, paths: usize) {
    println!("Code to show that if EURGBP negatively correlated with GBP interest rates, how does this change TV of strike-out put");
    println!("{:<15} | {:<15}", "Correlation", "TV (DOP)");
    println!("{:-<35}", "");

    for rho in [-0.5, 0.0, 0.5] {
        let tv = simulate_correlated_dop(s0, k, h, T, rd_base, rf, sigma_s, sigma_r, rho, paths);
        println!("{:<15.2} | {:<15.6}", rho, tv);
    }
}

fn simulate_correlated_dop(s0: f64, k: f64, h: f64, t: f64, rd_base: f64, rf: f64, sigma_s: f64, sigma_r: f64, rho: f64, paths: usize) -> f64 {
    let mut rng = thread_rng();
    let steps = 252;
    let dt = t / steps as f64;
    let mut total_payoff = 0.0;

    for _ in 0..paths {
        let mut s = s0;
        let mut rd = rd_base;
        let mut knocked_out = false;

        for _ in 0..steps {
            let z1: f64 = rng.sample(StandardNormal);
            let z2: f64 = rng.sample(StandardNormal);
            
            // Correlated random variables
            let dw_s = z1;
            let dw_r = rho * z1 + (1.0 - rho.powi(2)).sqrt() * z2;

            // Update GBP rate and EURGBP Spot
            rd += sigma_r * dt.sqrt() * dw_r;
            let mu = rd - rf;
            s *= ((mu - 0.5 * sigma_s.powi(2)) * dt + sigma_s * dt.sqrt() * dw_s).exp();

            if s <= h {
                knocked_out = true;
                break;
            }
        }

        if !knocked_out {
            let payoff = (k - s).max(0.0);
            total_payoff += payoff * (-rd * t).exp();
        }
    }
    total_payoff / paths as f64
}