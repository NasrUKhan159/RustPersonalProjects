use rand::prelude::*;
use rand_distr::{Normal, Distribution};
use std::f64::consts::PI;

struct Params {
    s0: f64, r: f64, sigma: f64, t1: f64, t2: f64, d: f64, k_mult: f64,
}

// Add this enum to handle the two types
enum OptionType {
    Call,
    Put,
}

pub fn main() {
    // Synthetic example of EURHUF forward start option
    // r = drift ~ 4% since ECB rates ~ 2% but Hungarian interest rates ~ 6%
    // Assume forward start date is in 3 months and final expiry is 1Y from now 
    // d: from strike = spot rate + d. 2 cases of call and put option
    let p = Params { s0: 388.0, r: 0.04, sigma: 0.12, t1: 0.25, t2: 1.0, d: 5.0, k_mult: 1.0 };
    let n_sims = 1_000;

    let val_custom = price_fwd_start(&p, n_sims, true, OptionType::Call);
    let val_std = price_fwd_start(&p, n_sims, false, OptionType::Put);
    
    println!("Forward Start (K = S_t1 + d): {:.4}", val_custom);
    println!("Forward Start (Standard K = S_t1): {:.4}", val_std);
    
    // Window Barrier vs Standard
    let b_mult = 1.2; // Barrier at 1.2 * S_t
    let (std_b, win_b) = price_barriers(&p, n_sims, b_mult, 0.5, 1.5);
    println!("Standard Knock-Out: {:.4}", std_b);
    println!("Window Knock-Out (0.5 to 1.5s): {:.4}", win_b);

    // Part 2: Back out forward volatility smile using option values
    // Synthetic eg of GBPJPY forward start option
    let s0 = 214.82;
    let r = 0.03; // interest rate differential (BOE rates around 4% but BoJ rates around 1%)
    let t1 = 0.25;
    let t2 = 1.25;

    // Simulated market data: (Strike Multiplier, Market Price)
    // Prices often reflect a 'smile' where OTM/ITM have higher implied vol
    let market_data = vec![
        (0.8, 48.50), // ITM: Higher IV (approx 16%) due to JPY tail risk
        (0.9, 32.10), // Slight ITM
        (1.0, 18.50), // ATM 
        (1.1, 9.20), // OTM
        (1.2, 4.10), // deep OTM
    ];
    println!("Backing out forward volatility smile for toy eg of GBPJPY fwd start option");
    println!("Strike (k) | Market Price | Implied Forward Vol");
    println!("-----------|--------------|--------------------");
    for (k, mkt_p) in market_data {
        let iv = back_out_implied_vol(mkt_p, s0, r, t1, t2, k);
        println!("{:>10.1} | {:>12.2} | {:>18.2}%", k, mkt_p, iv * 100.0);
    }
}

fn price_fwd_start(p: &Params, n: usize, custom: bool, option_type: OptionType) -> f64 {
    let mut rng = thread_rng();
    let norm = Normal::new(0.0, 1.0).unwrap();
    let mut total_payoff = 0.0;

    for _ in 0..n {
        let z1 = norm.sample(&mut rng);
        let z2 = norm.sample(&mut rng);
        
        // Simulating the spot at t1
        let s_t1 = p.s0 * ((p.r - 0.5 * p.sigma.powi(2)) * p.t1 + p.sigma * p.t1.sqrt() * z1).exp();
        
        // Simulating the spot at t2 starting from s_t1
        let s_t2 = s_t1 * ((p.r - 0.5 * p.sigma.powi(2)) * (p.t2 - p.t1) + p.sigma * (p.t2 - p.t1).sqrt() * z2).exp();
        
        let strike = if custom { s_t1 + p.d } else { s_t1 * p.k_mult };

        // Flexible payoff logic
        let payoff = match option_type {
            OptionType::Call => (s_t2 - strike).max(0.0),
            OptionType::Put  => (strike - s_t2).max(0.0),
        };
        
        total_payoff += payoff;
    }
    
    // Discount the average payoff back from t2 to today
    (total_payoff / n as f64) * (-p.r * p.t2).exp()
}

fn price_barriers(p: &Params, n: usize, b_mult: f64, w_start: f64, w_end: f64) -> (f64, f64) {
    let mut rng = thread_rng();
    let norm = Normal::new(0.0, 1.0).unwrap();
    let dt = 1.0 / 252.0; // Daily monitoring
    let steps = (p.t2 / dt) as usize;
    
    let mut pay_std = 0.0;
    let mut pay_win = 0.0;

    for _ in 0..n {
        let mut s = p.s0;
        let mut knocked_std = false;
        let mut knocked_win = false;
        let strike = p.s0 * p.k_mult;

        for step in 1..=steps {
            let t = step as f64 * dt;
            s *= ((p.r - 0.5 * p.sigma.powi(2)) * dt + p.sigma * dt.sqrt() * norm.sample(&mut rng)).exp();
            
            let barrier = s * b_mult; // Barrier as multiple of current spot
            if s >= barrier {
                knocked_std = true;
                if t >= w_start && t <= w_end { knocked_win = true; }
            }
        }
        if !knocked_std { pay_std += (s - strike).max(0.0); }
        if !knocked_win { pay_win += (s - strike).max(0.0); }
    }
    ((pay_std / n as f64) * (-p.r * p.t2).exp(), (pay_win / n as f64) * (-p.r * p.t2).exp())
}

/// Normal Cumulative Distribution Function
fn cdf(x: f64) -> f64 {
    0.5 * (1.0 + statrs::function::erf::erf(x / 2.0f64.sqrt()))
}

/// Normal Probability Density Function
fn pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Prices a standard Forward-Start Option (K = k * S_T1)
fn fwd_start_price_and_vega(s0: f64, r: f64, sigma: f64, t1: f64, t2: f64, k_mult: f64) -> (f64, f64) {
    let tau = t2 - t1;
    let sqrt_tau = tau.sqrt();
    let d1 = ((1.0 / k_mult).ln() + (r + 0.5 * sigma.powi(2)) * tau) / (sigma * sqrt_tau);
    let d2 = d1 - sigma * sqrt_tau;

    // Price formula for forward-start call
    let price = s0 * (cdf(d1) - k_mult * (-r * tau).exp() * cdf(d2));
    
    // Vega = S * sqrt(tau) * pdf(d1)
    let vega = s0 * sqrt_tau * pdf(d1);
    
    (price, vega)
}

/// Newton-Raphson solver to back out Implied Volatility
fn back_out_implied_vol(market_price: f64, s0: f64, r: f64, t1: f64, t2: f64, k_mult: f64) -> f64 {
    let mut sigma = 0.20; // Initial guess
    let epsilon = 1e-6;
    let max_iter = 100;

    for _ in 0..max_iter {
        let (price, vega) = fwd_start_price_and_vega(s0, r, sigma, t1, t2, k_mult);
        let diff = price - market_price;

        if diff.abs() < epsilon || vega.abs() < 1e-8 {
            break;
        }
        sigma -= diff / vega; // Newton step: sigma = sigma - f(sigma)/f'(sigma)

        // Safeguard: Avoid the newton-raphson method from giving negative volatility
        if sigma <= 0.0 {sigma = 0.0001;}
    }
    sigma
}