use statrs::distribution::{ContinuousCDF, Normal}; // cargo add statrs

#[derive(Clone, Debug)]
struct CurrencyPair {
    base: String,
    quote: String,
    spot: f64,
    vol: f64,
}

impl CurrencyPair {
    /// Normalises any pair to a USD-denominated rate (Price of 1 Unit of Base in USD).
    fn to_usd_basis(&self, bridge: Option<&CurrencyPair>) -> (f64, f64) {
        if self.quote == "USD" {
            (self.spot, self.vol)
        } else if self.base == "USD" {
            // Invert USD/XXX to get XXX/USD
            (1.0 / self.spot, self.vol) 
        } else {
            let b = bridge.expect("Bridge required for non-USD pairs");
            if b.base == self.quote && b.quote == "USD" {
                // (BASE/QUOTE) * (QUOTE/USD) = BASE/USD
                (self.spot * b.spot, (self.vol.powi(2) + b.vol.powi(2)).sqrt())
            } else if b.quote == self.quote && b.base == "USD" {
                // (BASE/QUOTE) / (USD/QUOTE) = BASE/USD
                (self.spot / b.spot, (self.vol.powi(2) + b.vol.powi(2)).sqrt())
            } else {
                panic!("Bridge {}/{} cannot link {}/{} to USD", b.base, b.quote, self.base, self.quote);
            }
        }
    }
}

fn kirk_spread_price(f1: f64, f2: f64, k: f64, t: f64, v1: f64, v2: f64, rho: f64, r: f64) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    let f3 = f2 + k;
    let v_spread = ((v1 * f1 / f3).powi(2) + (v2 * f2 / f3).powi(2) 
                    - 2.0 * rho * v1 * v2 * (f1 * f2 / f3.powi(2))).sqrt();
    
    let d1 = ((f1 / f3).ln() + 0.5 * v_spread.powi(2) * t) / (v_spread * t.sqrt());
    let d2 = d1 - v_spread * t.sqrt();
    
    (-r * t).exp() * (f1 * n.cdf(d1) - f3 * n.cdf(d2))
}

pub fn main() {
    // Shared parameters
    let (r, t, rho) = (0.05, 0.5, 0.6);

    // Helper closure to calculate and print ATM Kirk spread price
    let run_case = |label: &str, s1: f64, v1: f64, s2: f64, v2: f64| {
        // Option 2: Set K to the current spread (S1 - S2) to make it ATM
        let k_atm = s1 - s2;
        let price = kirk_spread_price(s1, s2, k_atm, t, v1, v2, rho, r);
        println!(
            "{:<25} | S1: {:.4}, S2: {:.4} | Spread: {:>7.4} | ATM Price: {:.6}", 
            label, s1, s2, s1 - s2, price
        );
    };

    println!("{:-<100}", "");
    println!("{:<25} | {:<21} | {:<15} | {:<10}", "Case", "Spot Prices (USD)", "Market Spread", "Kirk Price");
    println!("{:-<100}", "");

    // Case 1: USDHKD/USDSGD
    let c1_p1 = CurrencyPair { base: "USD".into(), quote: "HKD".into(), spot: 7.82, vol: 0.05 };
    let c1_p2 = CurrencyPair { base: "USD".into(), quote: "SGD".into(), spot: 1.34, vol: 0.08 };
    let (s1, v1) = c1_p1.to_usd_basis(None);
    let (s2, v2) = c1_p2.to_usd_basis(None);
    run_case("1: USDHKD / USDSGD", s1, v1, s2, v2);

    // Case 2: USDNOK / GBPUSD
    let c2_p1 = CurrencyPair { base: "USD".into(), quote: "NOK".into(), spot: 10.55, vol: 0.14 };
    let c2_p2 = CurrencyPair { base: "GBP".into(), quote: "USD".into(), spot: 1.27, vol: 0.10 };
    let (s1, v1) = c2_p1.to_usd_basis(None);
    let (s2, v2) = c2_p2.to_usd_basis(None);
    run_case("2: USDNOK / GBPUSD", s1, v1, s2, v2);

    // Case 3: AUDUSD / USDSEK
    let c3_p1 = CurrencyPair { base: "AUD".into(), quote: "USD".into(), spot: 0.66, vol: 0.12 };
    let c3_p2 = CurrencyPair { base: "USD".into(), quote: "SEK".into(), spot: 10.40, vol: 0.14 };
    let (s1, v1) = c3_p1.to_usd_basis(None);
    let (s2, v2) = c3_p2.to_usd_basis(None);
    run_case("3: AUDUSD / USDSEK", s1, v1, s2, v2);

    // Case 4: NZDUSD / EURPLN (with bridge)
    let c4_p1 = CurrencyPair { base: "NZD".into(), quote: "USD".into(), spot: 0.61, vol: 0.11 };
    let c4_p2 = CurrencyPair { base: "EUR".into(), quote: "PLN".into(), spot: 4.32, vol: 0.09 };
    let b4 = CurrencyPair { base: "USD".into(), quote: "PLN".into(), spot: 3.98, vol: 0.10 };
    let (s1, v1) = c4_p1.to_usd_basis(None);
    let (s2, v2) = c4_p2.to_usd_basis(Some(&b4));
    run_case("4: NZDUSD / EURPLN", s1, v1, s2, v2);

    // Case 5: AUDNZD / CHFJPY (double bridge)
    let c5_p1 = CurrencyPair { base: "AUD".into(), quote: "NZD".into(), spot: 1.08, vol: 0.07 };
    let b5_1 = CurrencyPair { base: "NZD".into(), quote: "USD".into(), spot: 0.61, vol: 0.11 };
    let c5_p2 = CurrencyPair { base: "CHF".into(), quote: "JPY".into(), spot: 171.0, vol: 0.09 };
    let b5_2 = CurrencyPair { base: "USD".into(), quote: "JPY".into(), spot: 151.0, vol: 0.12 };
    let (s1, v1) = c5_p1.to_usd_basis(Some(&b5_1));
    let (s2, v2) = c5_p2.to_usd_basis(Some(&b5_2));
    run_case("5: AUDNZD / CHFJPY", s1, v1, s2, v2);
}