use faer::{Mat, prelude::*};
use statrs::function::gamma::gamma;

pub struct FxOptionParams {
    pub s_max: f64, pub k: f64, pub t: f64, pub rd: f64, 
    pub rf: f64, pub sigma: f64, pub alpha: f64,
}

pub fn solve_fx_tfbs_final_stable(params: FxOptionParams, m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let x_min = (params.k / 10.0).ln();
    let x_max = params.s_max.ln();
    let dx = (x_max - x_min) / m as f64;
    let dt = params.t / n as f64;
    
    let s_grid: Vec<f64> = (0..=m).map(|i| (x_min + i as f64 * dx).exp()).collect();

    // PDE Coeffs
    let sigma2 = params.sigma.powi(2);
    let drift = (params.rd - params.rf) - 0.5 * sigma2;
    // Scale factor d = Gamma(2-alpha) * dt^alpha
    let d = dt.powf(params.alpha) * gamma(2.0 - params.alpha);

    // Weights b_j = (j+1)^(1-alpha) - j^(1-alpha)
    let b: Vec<f64> = (0..=n).map(|j| (j as f64 + 1.0).powf(1.0 - params.alpha) - (j as f64).powf(1.0 - params.alpha)).collect();

    let mut v = Mat::<f64>::zeros(m + 1, n + 1);
    for i in 0..=m { v[(i, 0)] = (s_grid[i] - params.k).max(0.0); }

    // Discretization coefficients for matrix A
    let alpha_coeff = d * (sigma2 / (2.0 * dx.powi(2)));
    let beta_coeff = d * (drift / (2.0 * dx));
    let gamma_coeff = d * params.rd;

    let main_diag = 1.0 + 2.0 * alpha_coeff + gamma_coeff;
    let upper_val = -(alpha_coeff + beta_coeff);
    let lower_val = -(alpha_coeff - beta_coeff);

    let mut a_matrix = Mat::<f64>::zeros(m - 1, m - 1);
    for i in 0..(m - 1) {
        a_matrix[(i, i)] = main_diag;
        if i > 0 { a_matrix[(i, i - 1)] = lower_val; }
        if i < m - 2 { a_matrix[(i, i + 1)] = upper_val; }
    }
    let lu = a_matrix.partial_piv_lu();

    // Time Stepping
    for step in 1..=n {
        let mut rhs = Mat::<f64>::zeros(m - 1, 1);
        let t_curr = step as f64 * dt;
        let v_upper = params.s_max * (-params.rf * t_curr).exp() - params.k * (-params.rd * t_curr).exp();

        for i in 1..m {
            // Correct L1 History: Sum_{j=1}^{step-1} (b_{j-1} - b_j) * V_{step-j} + b_{step-1} * V_0
            let mut history = 0.0;
            if step > 1 {
                for k in 1..step {
                    history += (b[k - 1] - b[k]) * v[(i, step - k)];
                }
            }
            history += b[step - 1] * v[(i, 0)];
            rhs[(i - 1, 0)] = history;
        }

        // Apply boundary condition to the last equation in the tridiagonal system
        rhs[(m - 2, 0)] -= upper_val * v_upper;

        let sol = lu.solve(&rhs);
        for i in 1..m { v[(i, step)] = sol[(i - 1, 0)]; }
        
        v[(0, step)] = 0.0; // Left boundary S -> 0
        v[(m, step)] = v_upper; // Right boundary S -> S_max
    }

    (s_grid, (0..=m).map(|i| v[(i, n)]).collect())
}