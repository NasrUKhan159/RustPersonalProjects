mod main2;

use main2::{mainCase2};

use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::time::Instant;

// 1. RAW EMA IMPLEMENTATION
// CustomEMA struct represents persistent state
// Each new Tick is an input that "updates" the state 
// (the EMA value) without needing to store entire history
// of prices in a database
struct CustomEma {
    alpha: f64,
    current_ema: Option<f64>,
}

impl CustomEma {
    fn new(period: usize) -> Self {
        Self {
            alpha: 2.0 / (period as f64 + 1.0),
            current_ema: None,
        }
    }

    fn next(&mut self, price: f64) -> f64 {
        let new_ema = match self.current_ema {
            Some(prev) => (price * self.alpha) + (prev * (1.0 - self.alpha)),
            None => price, // Initialise with first tick
        };
        self.current_ema = Some(new_ema);
        new_ema
    }
}

// 2. DATA STRUCTURES
#[derive(Debug, Clone)]
struct Tick {
    symbol: String,
    price: f64,
}

struct RiskManager {
    max_exposure: Decimal,
    current_exposure: Decimal,
}

impl RiskManager {
    fn can_execute(&self, amount: Decimal) -> bool {
        self.current_exposure + amount <= self.max_exposure
    }
}

#[tokio::main]
async fn main() {

    // this way of calling case 1 or case 2 is failing when we try to call case 2. Need to fix!

    let case1_execute: bool = false; // if false, run case 2 else run case 1.

    if (case1_execute){
        // BACKPRESSURE: Buffer capacity of 5 ensures producer waits if consumer is slow
        // the below line ensures the channel can only hold 5 messages
        // mpsc::channel is a hot observable (or a pipeline)
        println!("Running case 1...");
        let (tx, mut rx) = mpsc::channel::<Tick>(5);
        let mut risk_mgr = RiskManager {
            max_exposure: dec!(5000.0),
            current_exposure: dec!(0.0),
        };

        let mut ema_fast = CustomEma::new(9);
        let mut ema_slow = CustomEma::new(21);
        let mut last_cross_up = false;
        let start_time = Instant::now();

        // PRODUCER: Random Market Data (10ms intervals)
        // By using tokio::spawn, data ingestion remains indepedent of strategy processing
        // preventing slow execution from blocking market data feed
        tokio::spawn(async move {
            let mut price = 150.0; // Start in the middle
            let min_price = 149.95;
            let max_price = 150.05;
            let increment = 0.001; // Small steps to make it smooth
            let mut direction = 1.0; // 1.0 for up, -1.0 for down

            loop {
                price += increment * direction;

                // Reverse direction if bounds are hit
                if price >= max_price || price <= min_price {
                    direction *= -1.0;
                }

                let tick = Tick { symbol: "USD-JPY".into(), price };
                
                if tx.send(tick).await.is_err() { break; }
                
                sleep(Duration::from_millis(10)).await;
            }
        });

        // CONSUMER: Reactive Strategy logic (Slowed to 50ms to trigger backpressure)
        println!("Starting reactive engine... Monitoring crossovers.");
        
        // Interpretation of while loop: whenever an event arrives, perform these operations
        // This separates the business logic (calculate EMA, check risk) from how it gets data
        while let Some(tick) = rx.recv().await {
            // Simulate heavy computation (triggering backpressure)
            // the sleep call below intenttionally delays processing for 50ms per tick
            // while producer tries to run much faster than consumer, overall throughput
            // matches consumer's speed (roughly one tick every 50ms) preventing memory 
            // buildup or system overload since producer forced to wait until 5-slot buffer
            // is full.
            sleep(Duration::from_millis(50)).await;

            let f_val = ema_fast.next(tick.price);
            let s_val = ema_slow.next(tick.price);
            let current_cross_up = f_val > s_val;

            if current_cross_up && !last_cross_up {
                let trade_amt = dec!(1000.0);
                if risk_mgr.can_execute(trade_amt) {
                    risk_mgr.current_exposure += trade_amt;
                    println!("[{:?}] BUY  | Price: {:.2} | Exposure: {}", 
                        start_time.elapsed(), tick.price, risk_mgr.current_exposure);
                } else {
                    println!("[{:?}] RISK LIMIT | No Trade", start_time.elapsed());
                }
            } else if !current_cross_up && last_cross_up {
                risk_mgr.current_exposure = dec!(0.0);
                println!("[{:?}] SELL | Price: {:.2} | Position Cleared", 
                    start_time.elapsed(), tick.price);
            }

            last_cross_up = current_cross_up;
        }
    } else {
        println!("Running case 2...");
        mainCase2();
    } 
}