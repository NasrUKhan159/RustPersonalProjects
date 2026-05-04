// Delta: rate of change of option price w.r.t spot price
// Gamma: Rate of change of delta, theta: rate of change of option price w.r.t time
// Up/down factors: computed as u = exp{\sigma * \sqrt{\delta * t}} and d = 1/u 
// to ensure binomial tree recombines
// FX Probability: Risk-neutral prob p incorporates (r_{d} - r_{f}) to account for 
// "dividend yield" effect of foreign currency. 
// Corridor payoff: Unlike standard call/put, corridor payoff is binary (1 if inside range, 0 if out range)
// Bermudan logic: At each `exercise_step`, algo compares discounted expected future value
// against immediate payoff (whether spot currently in corridor or not)

struct Greeks {
    price: f64,
    delta: f64,
    gamma: f64,
    theta: f64,
}

pub fn main() {
    // E.g. with EURUSD FX Bermudan option
    let spot = 1.10;
    let low = 1.05;
    let high = 1.15;
    let rd = 0.05;
    let rf = 0.03;
    let vol = 0.10;
    let expiry = 1.0;
    let steps = 200; // Sufficient for convergence
    let exercise_steps = vec![50, 100, 150]; // Example Bermudan dates

    let result = calculate_stable_greeks(spot, low, high, rd, rf, vol, expiry, steps, &exercise_steps);

    println!("Price:  {:.6}", result.price);
    println!("Delta:  {:.6}", result.delta);
    println!("Gamma:  {:.6}", result.gamma);
    println!("Theta:  {:.6} (per day)", result.theta / 365.0);
}

fn price_bermudan_tree(
    s: f64, l: f64, h: f64, rd: f64, rf: f64, vol: f64, t: f64, 
    steps: usize, exercise_steps: &[usize]
) -> f64 {
    // 1. Guard against non-positive time (Prevents NaN in sqrt and dt)
    if t <= 1e-10 {
        return if s >= l && s <= h { 1.0 } else { 0.0 };
    }

    let dt = t / steps as f64;
    let u = (vol * dt.sqrt()).exp();
    let d = 1.0 / u;
    
    // 2. Risk Neutral Probability (Garman-Kohlhagen)
    let p = (((rd - rf) * dt).exp() - d) / (u - d);
    let disc = (-rd * dt).exp();

    // 3. Pre-calculate spot prices at terminal nodes to avoid powf inside loops
    let mut values: Vec<f64> = vec![0.0; steps + 1];
    for i in 0..=steps {
        let spot_t = s * u.powi((steps as i32) - 2 * (i as i32));
        values[i] = if spot_t >= l && spot_t <= h { 1.0 } else { 0.0 };
    }

    // 4. Backward induction
    for step in (0..steps).rev() {
        for i in 0..=step {
            let continuation = disc * (p * values[i] + (1.0 - p) * values[i + 1]);
            
            if exercise_steps.contains(&step) {
                let spot_now = s * u.powi((step as i32) - 2 * (i as i32));
                let exercise_val = if spot_now >= l && spot_now <= h { 1.0 } else { 0.0 };
                values[i] = continuation.max(exercise_val);
            } else {
                values[i] = continuation;
            }
        }
    }
    values[0]
}

fn calculate_stable_greeks(
    s: f64, l: f64, h: f64, rd: f64, rf: f64, vol: f64, t: f64, 
    steps: usize, exercise_steps: &[usize]
) -> Greeks {
    // Bump parameters
    let ds = 0.01 * s; // 1% bump for digital stability
    let dt_bump = 1.0 / 365.0;

    let p_mid = price_bermudan_tree(s, l, h, rd, rf, vol, t, steps, exercise_steps);
    let p_up = price_bermudan_tree(s + ds, l, h, rd, rf, vol, t, steps, exercise_steps);
    let p_down = price_bermudan_tree(s - ds, l, h, rd, rf, vol, t, steps, exercise_steps);
    
    // Ensure theta bump doesn't go negative
    let t_theta = if t > dt_bump { t - dt_bump } else { 0.0 };
    let p_theta = price_bermudan_tree(s, l, h, rd, rf, vol, t_theta, steps, exercise_steps);

    Greeks {
        price: p_mid,
        delta: (p_up - p_down) / (2.0 * ds),
        gamma: (p_up - 2.0 * p_mid + p_down) / (ds * ds),
        theta: (p_theta - p_mid), // Change in value per day
    }
}