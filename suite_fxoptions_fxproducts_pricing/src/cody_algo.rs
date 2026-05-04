use std::f64::consts::SQRT_2;

/// Cody's rational approximation for the Complementary Error Function (erfc)
fn erfc_cody(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    // Rational coefficients for high precision
    let ans = t * (-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + 
              t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + 
              t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + 
              t * 0.17087277)))))))));
    if x >= 0.0 { ans.exp() } else { 2.0 - ans.exp() }
}

/// Normal CDF using Cody's erfc
fn norm_cdf(x: f64) -> f64 {
    if x >= 0.0 { 1.0 - 0.5 * erfc_cody(x / SQRT_2) }
    else { 0.5 * erfc_cody(-x / SQRT_2) }
}

struct FXInstalment {
    spot: f64,
    strike: f64,
    vol: f64,
    r_d: f64, // Domestic rate
    r_f: f64, // Foreign rate
    instalments: Vec<(f64, f64)>, // (Time, Amount)
}

impl FXInstalment {
    /// Value at t=0 using Stochastic Dynamic Programming (Backward Induction)
    fn value_at_0(&self, grid_size: usize) -> f64 {
        let t_max = self.instalments.last().unwrap().0;
        let mut spot_grid: Vec<f64> = (0..grid_size)
            .map(|i| self.spot * (0.5 + i as f64 / grid_size as f64))
            .collect();

        // 1. Terminal Payoff at Maturity
        let mut values: Vec<f64> = spot_grid.iter()
            .map(|&s| (s - self.strike).max(0.0))
            .collect();

        // 2. Backward Induction through instalment dates
        for i in (0..self.instalments.len()).rev() {
            let (t_curr, inst_amt) = self.instalments[i];
            let t_prev = if i == 0 { 0.0 } else { self.instalments[i-1].0 };
            let dt = t_curr - t_prev;

            let mut next_values = vec![0.0; grid_size];
            for j in 0..grid_size {
                // Continuation Value: Expected value under Risk-Neutral Measure
                // In a simple grid, this uses the transition density (Normal via Cody)
                let expected_value = self.calculate_expectation(spot_grid[j], &spot_grid, &values, dt);
                
                // SDP Decision: Pay instalment to keep option, or let it lapse (value = 0)
                next_values[j] = (expected_value - inst_amt).max(0.0);
            }
            values = next_values;
        }

        // Interpolate to find value at the initial spot price
        values[grid_size / 2] // Simplified: assumes initial spot is center of grid
    }

    fn calculate_expectation(&self, s: f64, grid: &[f64], values: &[f64], dt: f64) -> f64 {
        // Simplified expectation calculation using the transition density
        // In production, use a transition matrix or Gaussian quadrature
        let mut sum = 0.0;
        let drift = (self.r_d - self.r_f - 0.5 * self.vol.powi(2)) * dt;
        let std_dev = self.vol * dt.sqrt();

        for k in 0..grid.len() {
            let z = ((grid[k] / s).ln() - drift) / std_dev;
            let prob = norm_cdf(z); // Precision from Cody algorithm
            sum += values[k] * prob;
        }
        sum / grid.len() as f64 // Normalized
    }
}

pub fn main() {
    let option = FXInstalment {
        // Example case of EURUSD option
        spot: 1.20,
        strike: 1.22,
        vol: 0.15,
        r_d: 0.03,
        r_f: 0.01,
        instalments: vec![(0.5, 0.02), (1.0, 0.02)],
    };

    println!("Cody Algo - Option Value at T=0 for example EURUSD option (payoff per 1 unit of EUR notional): {}", option.value_at_0(100));
}