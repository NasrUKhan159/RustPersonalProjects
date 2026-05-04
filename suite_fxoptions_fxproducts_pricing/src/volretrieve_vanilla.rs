use statrs::distribution::{Continuous, Normal, ContinuousCDF};

/// Black-Scholes (Garman-Kohlhagen) FX Option Price and Vega
fn fx_option_stats(s: f64, k: f64, t: f64, rd: f64, rf: f64, sigma: f64, is_call: bool) -> (f64, f64) {
    let n = Normal::new(0.0, 1.0).unwrap();
    let sqrt_t = t.sqrt();
    let d1 = ( (s / k).ln() + (rd - rf + 0.5 * sigma.powi(2)) * t ) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;

    let price = if is_call {
        s * (-rf * t).exp() * n.cdf(d1) - k * (-rd * t).exp() * n.cdf(d2)
    } else {
        k * (-rd * t).exp() * n.cdf(-d2) - s * (-rf * t).exp() * n.cdf(-d1)
    };

    // Vega is the derivative of price w.r.t sigma
    let vega = s * (-rf * t).exp() * n.pdf(d1) * sqrt_t;

    (price, vega)
}

/// Implied Volatility using Newton's Method with Saddle Point Guess
pub fn implied_volatility(
    market_price: f64,
    s: f64,
    k: f64,
    t: f64,
    rd: f64,
    rf: f64,
    is_call: bool,
) -> Option<f64> {
    // Initial guess at the saddle point (inflection point) where d1*d2 = 0
    // This is approximately where the Vega is maximized.
    let mut sigma = (2.0 * ((s / k).ln().abs() + (rd - rf).abs() * t) / t).sqrt().max(0.1);
    
    let max_iter = 100;
    let tolerance = 1e-8;

    for _ in 0..max_iter {
        let (price, vega) = fx_option_stats(s, k, t, rd, rf, sigma, is_call);
        let diff = price - market_price;

        if diff.abs() < tolerance {
            return Some(sigma);
        }

        // Avoid division by zero if Vega is too small
        if vega < 1e-10 { break; }

        let next_sigma = sigma - diff / vega;
        
        // Ensure sigma stays positive
        sigma = next_sigma.max(1e-6);
    }
    None
}

pub fn main(){
    // Some synthetic values for GBPEUR and AUDNZD options and what their implied vol would be
    // Synthetic Example 1: GBPEUR Call Option
    let market_price_1 = 0.0076; // A 1 month call option assumed to be priced at 0.0076 EUR per GBP
    let s_1 = 1.1490; 
    let k_1 = 1.15;
    let t_1 = 1.0/12.0; // 1M option
    let rd_1 = 0.02;
    let rf_1 = 0.0375;
    let is_call_1 = true;
    // Synthetic Example 2: AUDNZD Call Option
    let market_price_2 = 0.02; // A 3 month put option assumed to be priced at 0.02 NZD per AUD
    let s_2 = 1.22; 
    let k_2 = 1.22; 
    let t_2 = 3.0/12.0; // 3M option
    let rd_2 = 0.035;
    let rf_2 = 0.0425;
    let is_call_2 = false;
    let implied_vol_gbpeur = implied_volatility(market_price_1, s_1, k_1, t_1, rd_1, rf_1, is_call_1);
    let implied_vol_audnzd = implied_volatility(market_price_2, s_2, k_2, t_2, rd_2, rf_2, is_call_2);
    match implied_vol_gbpeur {
        Some(vol) => println!("The implied vol obtained via saddle pt method for eg GBPEUR option {:.2}%", vol * 100.0),
        None => println!("Could not find a valid implied volatility for GBPEUR eg.")
    }
        match implied_vol_audnzd {
        Some(vol) => println!("The implied vol obtained via saddle pt method for eg AUDNZD option {:.2}%", vol * 100.0),
        None => println!("Could not find a valid implied volatility for AUDNZD eg.")
    }
}