/* Model FX spot prices for EURUSD from 16-17 Dec using Kalman Filter
 Modelling FX spot using Kalman filter requires defining state-space model
 The state equation describes dynamics of the underlying, unobservable system
 state (e.g. "true" FX rate and an unobserved drift/volatility). The
 measurement equation relates observed market price to system state where
 observed price is a noisy measurement of true state. 
 The Kalman filter is an optimal estimator that provides both an estimate of 
 the current state and a prediction of the future state by recursively combining 
 predictions from model with new measurements.
 Data Source: Google Finance
*/

use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use kalman_filters::KalmanFilterBuilder;

// Define a struct that only includes the columns you need.
// The field names are matched with the CSV header names by default.

#[derive(Debug, Deserialize)]
struct EurUsdOnly {
    #[serde(rename = "EURUSD")]
    value: f64,
}

pub fn read_csv() -> Vec<f64> {
    let Ok(file) = File::open("spot_rates.csv") else {
        return Vec::new(); // Return empty list if file doesn't exist
    };
    
    let mut rdr = csv::Reader::from_reader(file);

    rdr.deserialize::<EurUsdOnly>()
        .filter_map(|result| result.ok()) // Skips any rows that fail to parse
        .map(|record| record.value)
        .collect()
}

pub fn model_kalman_filter(fx_data: Vec<f64>) -> anyhow::Result<()> {
    if fx_data.is_empty() {
        return Err(anyhow::anyhow!("Input FX data is empty"));
    }

    let drift = 1.0;

    // Create a 1D tracker (1 state: price, 1 measurement: spot price)
    let mut kf = KalmanFilterBuilder::new(1, 1)
        .initial_state(vec![fx_data[0]]) // Initial price is first data point
        .initial_covariance(vec![1e-4]) // Small initial uncertainty
        .transition_matrix(vec![drift])   // F matrix measuring state transition: price at t 
        // is price at t-1
        .process_noise(vec![1e-5])      // Process noise Q (measuring how much the underlying
        // price can change each step)
        .observation_matrix(vec![1.0])  // Observation matrix H: measure price directly
        .measurement_noise(vec![5e-5])  // Measurement noise R (measuring uncertainty in 
        // observed price)
        .build()
        .map_err(|e| anyhow::anyhow!(format!("{:?}", e)))?;

    println!("Step | Observation | Filtered State | Predicted Next State");
    println!("------------------------------------------------------");

    for (i, &measurement) in fx_data.iter().enumerate() {
        if i > 0 {
            // 1. Predict the state based on the previous step
            kf.predict();

            // 2. Update the prediction with the current observation
            kf.update(&[measurement])
                .map_err(|e| anyhow::anyhow!(format!("{:?}", e)))?;

            // 3. Retrieve the updated state (the filtered price)
            let filtered_state = kf.state()[0];

            // 4. Calculate Predicted Next State (Option 1)
            // Since our transition matrix is 1.0, the prediction for 'i+1' 
            // is simply the current filtered state. So integrated position
            // logic directly.
            let predicted_next = filtered_state * drift; 

            println!(
                "{:4} | {:11.4} | {:14.4} | {:19.4}",
                i + 1,
                measurement,
                filtered_state,
                predicted_next
            );
        }
    }
    // Output the covariance of each prediction as well
    // the covariance measures the confidence in the filter
    // if the covariance shrinks and stabilises, suggesting that
    // the filtered state is a more certain estimate of the true price
    for (i, &measurement) in fx_data.iter().enumerate() {
        if i > 0 {
            // 1. Predict the state based on the previous step
            kf.predict();

            // 2. Update the prediction with the current observation
            kf.update(&[measurement])
                .map_err(|e| anyhow::anyhow!(format!("{:?}", e)))?;

            let state = kf.state();
            let covariance = kf.covariance();

            println!(
                "Step: {} | Price: {:.4} | Covariance: {:.4e}",
                i + 1,
                state[0],      // The filtered price
                covariance[0]  // The uncertainty/variance of that price
            );
        }
    }
    Ok(())
}

fn main() {
    let fx_data = read_csv();
    model_kalman_filter(fx_data);
}
// Quant traders use predicted next state as best guess 
// for next day's price (which in this case would be best guess for the next
// half an hour's EURUSD spot rate). If actual price next day is consistently
// higher than prediction, indicates bullish trend. A trading
// strategy would be to buy when actual price is much lower than Kalman prediction
// assuming price is temporarily low due to noise.
// NB: Extension could be to implement adaptive Kalman filter where we dynamically
// update Q or R, and increasing covariance would suggest market has entered high
// volatility regime where past data is less predictive.