Purpose of academic personal project: Build a reactive 
programming codebase for a finance and trading application.
In this, data flows through asynchronous, non-blocking streams.
In this code, backpressure management is also accounted for 
(handled situation where data production rates exceed consumption
rates, ensuring system remains stable and responsive under stress).
The example uses Tokio's mpsc (multi-producer, single-consumer) channels
which passes messages between concurrent tasks in a non-blocking, event-driven
manner. Output from code is a continuous stream of console output simulating
real-time market activity. Since system is asynchronous, pub-sub is happening
concurrently. After ingesting data, a "strategy" task computes crossovers and a
"risk/execution" layer validates trades before execution.

Different cases have been modelled:
Case 1 - main.rs: Fluctuating synthetic USD-JPY spot between 149.95 and 150.05
So we see the system cycling between hitting the limit and clearing it. The algo
is a simple EMA crossover. In such a narrow range, the EMAs will stay close together
leading to whipsaw trades where we buy/sell almost immediately where one can lose 
money on spread/slippage in a real environment.
Case 2: main2.rs - avoid whipsaw in Case 1 using a hysteresis/threshold. Signals 
only trigger if the difference between the fast and slow EMA is greater than the 
pre-defined threshold (0.05). Minor "noisy" crossovers within this deadband are 
ignored, preventing whipsaw trades. Extension to multi-symbols made.

Extensions to this:
1. Use Rust's Result and Option types to integrate failure recovery
mechanisms e.g. timeouts, circuit breakers.
2. Show trading updates in a UI using a model-view-model framework.
3. Multi-symbol strategy engine: manage simultaneous strategies for multiple symbols.
Approach: Change the consumer loop to handle a dynamic map (e.g., HashMap<String, StrategyState>). 
Each message would need to contain its symbol, and the consumer would look up and update the specific 
EMA/Risk state for that symbol. Key Libraries: std::collections::HashMap, tokio::sync::broadcast (if 
all subscribers need all data).
4. Backtesting framework: Validation of strategies against historical data:
Approach: Replace the tokio::spawn producer with a synchronous function that reads historical data (e.g., 
from a CSV file using the csv and serde crates) and feeds it into the same consumer logic. You can use 
the same while let Some(tick) = ... loop.
Key Insight: Because your strategy logic is decoupled from the data source by the channel interface, 
it becomes easily testable.
5. Asynchronous logging and metrics: 
Approach: Spawn a dedicated "Logging Service" task. Use a new mpsc::channel to send trade execution 
messages (the log data) to this service. This ensures the main trading logic thread never blocks 
waiting for slow I/O operations (like disk writes or database calls).
Key Libraries: csv crate for writing files, log or tracing for structured logging.
6. Configurable strategy parameters: Hardcoding EMA periods (9 and 21) or risk limits is impractical.
Approach: Load parameters from a configuration file (e.g., config.toml or environment variables) when 
the application starts. Pass these values to the CustomEma::new() and RiskManager structs upon 
initialization. Key Libraries: config or clap crates.
7. Strategy Trait and Plug-ins: abstracting the strategy logic behind a Trait allows you to easily 
swap algorithms without rewriting the main event loop. Approach: Define a trait Strategy with a method 
like fn process_tick(&mut self, tick: Tick) -> Option<Signal>. The main loop just calls this method, 
allowing you to easily implement and test EMA Crossover, RSI, MACD, etc.