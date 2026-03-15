use std::f64::consts::E;

#[derive(Debug)]
pub struct ASConfig {
    pub gamma: f64,    // Risk aversion (0.01 - 0.1)
    pub sigma: f64,    // Volatility (standard deviation of mid-price)
    pub kappa: f64,    // Order book liquidity/density
    pub dt: f64,       // Time horizon (remaining session time, e.g., 1.0)
}

pub struct Quote {
    pub bid: f64,
    pub ask: f64,
    pub reservation_price: f64,
}

impl ASConfig {
    /// Calculates the optimal bid/ask quotes based on current inventory
    pub fn calculate_quote(&self, mid_price: f64, inventory: f64) -> Quote {
        // 1. Calculate Reservation Price (r)
        // r = s - q * gamma * sigma^2 * dt
        let reservation_price = mid_price - (inventory * self.gamma * self.sigma.powi(2) * self.dt);

        // 2. Calculate Optimal Spread (s)
        // spread = gamma * sigma^2 * dt + (2/gamma) * ln(1 + gamma/kappa)
        let spread = (self.gamma * self.sigma.powi(2) * self.dt) 
                     + (2.0 / self.gamma) * (1.0 + self.gamma / self.kappa).ln();

        let half_spread = spread / 2.0;

        Quote {
            bid: reservation_price - half_spread,
            ask: reservation_price + half_spread,
            reservation_price,
        }
    }
}

fn main() {
    let engine = ASConfig {
        gamma: 0.1,    // Moderate risk aversion
        sigma: 0.0002, // Low FX volatility (e.g., 2 pips)
        kappa: 1.5,    // Liquidity parameter
        dt: 1.0,       // Standardised time unit
    };

    let mid_price = 1.0850; // EUR/USD mid
    let inventory = 500_000.0; // Long 500k EUR

    let quote = engine.calculate_quote(mid_price, inventory);

    println!("Mid: {:.5}", mid_price);
    println!("Reservation: {:.5}", quote.reservation_price);
    println!("Quote: {:.5} / {:.5}", quote.bid, quote.ask);
}