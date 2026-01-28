use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::fs;
use serde::Deserialize;

// CustomEma implementation remains the same as case 1
struct CustomEma {
    alpha: f64,
    current_ema: Option<f64>,
}
impl CustomEma {
    fn new(period: usize) -> Self { /* ... (same as case 1) ... */
        Self { alpha: 2.0 / (period as f64 + 1.0), current_ema: None }
    }
    fn next(&mut self, price: f64) -> f64 { /* ... (same as case 1) ... */
        let new_ema = match self.current_ema { Some(prev) => (price * self.alpha) + (prev * (1.0 - self.alpha)), None => price, };
        self.current_ema = Some(new_ema);
        new_ema
    }
}

// Tick struct remains the same as case 1
#[derive(Debug, Clone)]
struct Tick {
    symbol: String,
    price: f64,
}

// Strategy State for each symbol
struct SymbolState {
    ema_fast: CustomEma,
    ema_slow: CustomEma,
    last_cross_up: bool,
    exposure: Decimal, // Tracks exposure for this specific symbol
}

#[tokio::main]
pub async fn mainCase2() -> Result<(), Box<dyn std::error::Error>> {
    // 1. LOAD CONFIG FROM JSON
    let config_data = fs::read_to_string("config.json")?;
    let initial_prices: HashMap<String, f64> = serde_json::from_str(&config_data)?;
    let symbols: Vec<String> = initial_prices.keys().cloned().collect();

    let (tx, mut rx) = mpsc::channel::<Tick>(100);
    let start_time = Instant::now();
    let threshold = 0.01; // Narrow threshold for smaller price units

    type SharedState = Arc<Mutex<HashMap<String, SymbolState>>>;
    let strategy_state: SharedState = Arc::new(Mutex::new(HashMap::new()));

    // 2. PRODUCER: Uses unique starting prices from JSON
    let tx_clone = tx.clone();
    tokio::spawn(async move {
        let mut current_prices = initial_prices;
        loop {
            for (symbol, price) in current_prices.iter_mut() {
                // Random walk adjusted for smaller price scales
                *price += (rand::random::<f64>() - 0.5) * 0.01; 
                let _ = tx_clone.send(Tick { symbol: symbol.clone(), price: *price }).await;
            }
            sleep(Duration::from_millis(10)).await;
        }
    });

    // 3. CONSUMER: Reactive Strategy with JSON-derived Symbols
    println!("Reactive engine live. Tracking: {:?}", symbols);
    while let Some(tick) = rx.recv().await {
        let mut state_map = strategy_state.lock().unwrap();
        let state = state_map.entry(tick.symbol.clone()).or_insert_with(|| SymbolState {
            ema_fast: CustomEma::new(9),
            ema_slow: CustomEma::new(21),
            last_cross_up: false,
            exposure: rust_decimal_macros::dec!(0.0),
        });

        let f_val = state.ema_fast.next(tick.price);
        let s_val = state.ema_slow.next(tick.price);
        let diff = f_val - s_val;

        if diff > threshold && !state.last_cross_up {
            state.exposure += rust_decimal_macros::dec!(1.0);
            println!("[{:?}] BUY  {} | Price: {:.4}", start_time.elapsed(), tick.symbol, tick.price);
            state.last_cross_up = true;
        } else if diff < -threshold && state.last_cross_up {
            state.exposure = rust_decimal_macros::dec!(0.0);
            println!("[{:?}] SELL {} | Price: {:.4}", start_time.elapsed(), tick.symbol, tick.price);
            state.last_cross_up = false;
        }
    }
    Ok(())
}