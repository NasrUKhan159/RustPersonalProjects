use std::f64::consts::PI;
use nalgebra::{Matrix3, Vector3};

struct FXOptionParams {
    s: f64,    // Spot exchange rate (Domestic/Foreign)
    k: f64,    // Strike price
    t: f64,    // Time to maturity
    rd: f64,   // Domestic interest rate
    rf: f64,   // Foreign interest rate
    v: f64,    // Implied volatility
}

fn d1(p: &FXOptionParams) -> f64 {
    ((p.s / p.k).ln() + (p.rd - p.rf + 0.5 * p.v.powi(2)) * p.t) / (p.v * p.t.sqrt())
}

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x.powi(2)).exp() / (2.0 * PI).sqrt()
}

// FX Vanna: sensitivity of Vega to Spot (dVega/dSpot)
fn vanna_fx(p: &FXOptionParams) -> f64 {
    let d1_val = d1(p);
    let d2_val = d1_val - p.v * p.t.sqrt();
    let vega = p.s * (-p.rf * p.t).exp() * p.t.sqrt() * norm_pdf(d1_val);
    (vega / p.s) * (1.0 - d1_val / (p.v * p.t.sqrt()))
}

// FX Volga: sensitivity of Vega to Vol (dVega/dVol)
fn volga_fx(p: &FXOptionParams) -> f64 {
    let d1_val = d1(p);
    let d2_val = d1_val - p.v * p.t.sqrt();
    let vega = p.s * (-p.rf * p.t).exp() * p.t.sqrt() * norm_pdf(d1_val);
    vega * d1_val * d2_val / p.v
}

pub fn main() {
    // Eg values with EURUSD FX option
    let p = FXOptionParams { s: 1.10, k: 1.10, t: 0.5, rd: 0.03, rf: 0.01, v: 0.12 };
    println!("FX Vanna: {:.6}, FX Volga: {:.6}", vanna_fx(&p), volga_fx(&p));

    // Example Greek values for FX Options
    let atm = [0.40, 0.00, 0.05];    // High Vega, ~0 Vanna/Volga
    let c25 = [0.35, 0.08, 0.12];    // Positive Vanna (Skew)
    let p25 = [0.35, -0.08, 0.12];   // Negative Vanna
    let target = [1.0, 0.2, 0.5];    // Exotic portfolio to hedge

    match solve_hedge_weights(atm, c25, p25, target) {
        Some(weights) => {
            println!("Hedge Weights:");
            println!("ATM Option: {:.4}", weights[0]);
            println!("25D Call:   {:.4}", weights[1]);
            println!("25D Put:    {:.4}", weights[2]);
        }
        None => println!("The system is singular and cannot be solved."),
    }
}

/// Calculates the hedge weights (w1, w2, w3) for ATM, 25D Call, and 25D Put
/// to match a target portfolio's Vega, Vanna, and Volga.
fn solve_hedge_weights(
    atm_greeks: [f64; 3],  // [Vega, Vanna, Volga]
    call_greeks: [f64; 3], // [Vega, Vanna, Volga]
    put_greeks: [f64; 3],  // [Vega, Vanna, Volga]
    target: [f64; 3],      // [Target Vega, Target Vanna, Target Volga]
) -> Option<Vector3<f64>> {
    // Construct the matrix A (Greeks of the hedging instruments)
    // Rows: Vega, Vanna, Volga
    // Columns: ATM, 25DC, 25DP
    let a = Matrix3::new(
        atm_greeks[0], call_greeks[0], put_greeks[0], // Vega row
        atm_greeks[1], call_greeks[1], put_greeks[1], // Vanna row
        atm_greeks[2], call_greeks[2], put_greeks[2], // Volga row
    );

    let b = Vector3::new(target[0], target[1], target[2]);

    // Solve for w using LU decomposition
    a.lu().solve(&b)
}