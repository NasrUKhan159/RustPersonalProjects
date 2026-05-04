use core::num;

use rand::prelude::*;
use rand_distr::StandardNormal;

/// Values an FX forward with a decreasing notional schedule using path-dependent Monte-Carlo
pub fn price_amortizing_fx_forward(
    s0: f64,
    strike: f64,
    notionals: &[f64],
    times: &[f64],
    r_dom: f64,
    r_for: f64,
    vol: f64,
    num_sims: usize,
) -> f64 {
    let mut rng = thread_rng();
    let mut total_pv = 0.0;

    for _ in 0..num_sims {
        let mut path_pv = 0.0;
        let mut current_spot = s0;
        let mut last_t = 0.0;

        for (i, &t) in times.iter().enumerate() {
            let dt = t - last_t;
            let z: f64 = rng.sample(StandardNormal);
            
            // Step from the CURRENT spot to the next time point
            // This ensures the price follows a continuous path
            current_spot *= ((r_dom - r_for - 0.5 * vol * vol) * dt 
                            + vol * dt.sqrt() * z).exp();
            
            let payoff = (current_spot - strike) * notionals[i];
            path_pv += payoff * (-r_dom * t).exp();
            
            last_t = t;
        }
        total_pv += path_pv;
    }

    total_pv / num_sims as f64
}

/// Values a forward that renews if spot > strike at the first maturity.
pub fn price_autorenewal_fx_forward(
    s0: f64,
    k: f64,
    t1: f64,          // First expiry
    t2: f64,          // Second expiry if renewed
    r_dom: f64,
    r_for: f64,
    vol: f64,
    num_sims: usize,
) -> f64 {
    let mut rng = thread_rng();
    let mut total_payoff = 0.0;

    for _ in 0..num_sims {
        let z1: f64 = rng.sample(StandardNormal);
        let s1 = s0 * ((r_dom - r_for - 0.5 * vol * vol) * t1 + vol * t1.sqrt() * z1).exp();
        
        let mut path_value = (s1 - k) * (-r_dom * t1).exp();
        
        // Renewal logic: if ITM at T1, add the discounted payoff from T2
        if s1 > k {
            let z2: f64 = rng.sample(StandardNormal);
            let s2 = s1 * ((r_dom - r_for - 0.5 * vol * vol) * (t2 - t1) + vol * (t2 - t1).sqrt() * z2).exp();
            path_value += (s2 - k) * (-r_dom * t2).exp();
        }
        total_payoff += path_value;
    }
    total_payoff / num_sims as f64
}

/// Values an FX Time Option where exercise is allowed within [t_start, t_end].
/// 
/// # Arguments
/// * `t_start`: Year fraction when the exercise window opens
/// * `t_end`: Year fraction when the option expires
pub fn price_fx_time_option(
    s0: f64,
    k: f64,
    t_start: f64,
    t_end: f64,
    r_dom: f64,
    r_for: f64,
    vol: f64,
    steps: usize,
) -> f64 {
    let dt = t_end / steps as f64;
    let u = (vol * dt.sqrt()).exp();
    let d = 1.0 / u;
    let p = (((r_dom - r_for) * dt).exp() - d) / (u - d);
    let disc = (-r_dom * dt).exp();

    // Terminal values at t_end
    let mut values = vec![0.0; steps + 1];
    for j in 0..=steps {
        let st = s0 * u.powi(j as i32) * d.powi((steps - j) as i32);
        values[j] = (st - k).max(0.0);
    }

    // Backward induction
    for i in (0..steps).rev() {
        let t = i as f64 * dt;
        for j in 0..=i {
            let continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j]);
            if t >= t_start {
                let st = s0 * u.powi(j as i32) * d.powi((i - j) as i32);
                values[j] = continuation.max(st - k); // Early exercise if in window
            } else {
                values[j] = continuation;
            }
        }
    }
    values[0]
}

pub fn main(){

    // Assume an AUDUSD option
    let s0 = 0.7175;
    let k = 0.7;
    let r_dom = 0.0325; // US fed funds rate
    let r_for = 0.04; // AUD interest rate
    let t1: f64 = 1.0/12.0; // first expiry set to one month
    let t2: f64 = 1.0/4.0; // second expiry set to 3M
    let vol = 0.1; // assume short term volatility is 10%
    let num_sims = 100000; 
    // we need number of sims to be big for amortising fwd because in the e.g. 12-leg instrument, we need a large number of trials
    // for MC simulation
    let num_steps = 30; 
    // In amortising FX forward, `notionals` represent portion of contract being paid over time
    // And the `times` represent the schedule of future cash flows (e.g. monthly loan repayments or quarterly supply payments)
    // Simple case: Assume we are paying off AUD 200k each month across a year
    // Autorenewal FX forward: Assume first expiry in 1M and second expiry in 3M.
    // Time option: t_start = 0.0 (window immediately), t_end = 0.75 (9 months from now)
    let t_start = 0.0;
    let t_end = 0.75;
    let notionals= vec![200000.0; 12];
    let total_notional: f64 = notionals.iter().sum();
    let times = (1..=12).map(|m| m as f64 / 12.0).collect::<Vec<_>>();
    let amort_fx_fwd = price_amortizing_fx_forward(s0, k, &notionals, &times, r_dom, r_for, vol, num_sims);
    let autorenew_fx_fwd = price_autorenewal_fx_forward(s0, k, t1, t2, r_dom, r_for, vol, num_sims) * total_notional;
    let time_option = price_fx_time_option(s0, k, t_start, t_end, r_dom, r_for, vol, num_steps) * total_notional;
    println!("Value of AUDUSD amortizing FX forward for AUD 2.4M total notional contract: {}", amort_fx_fwd);
    // Understand values of all different products and see if they are right or wrong.
    println!("Value of AUDUSD autorenewal FX forward for AUD 2.4M total notional contract: {}", autorenew_fx_fwd);
    println!("Value of AUDUSD time option for AUD 2.4M total notional contract: {}", time_option);
}