/* Solve the Time-Fractional Black-Scholes (tfBS) equation to price FX options using 
 the L1 finite difference method. Black Scholes for pricing FX options is the 
 Garman-Kohlhagen framework, model, the foreign interest rate effectively acts as a "continuous 
 dividend yield". The fractional PDE governs the exchange rate dynamics with non-local memory effects, 
 where \(\alpha \) represents the fractional order
 The upper boundary condition in solving the PDE uses the discounted spot and strike price adjusted for both
 domestic and foreign rates. 
 The time fractional Black Scholes equation is expressed as:
 \frac{\partial^{\alpha}V}{\partial t^{\alpha}} + (r - q)S\frac{\partial V}{\partial S} + 
 \frac{1}{2}\sigma^{2}S^{2}\frac{\partial^{2}V}{\partial S^{2}} - rV = 0
 The L1 scheme approximates the Caputo time-fractional derivative at t_n as:
 \frac{\partial^{\alpha}V}{\partial t^{\alpha}} \approx \frac{\delta t^{-\alpha}}{\gamma(2 - \alpha)} *
 \sum_{j=0}^{n-1} b_{j}(V_{i}^{n-j} - V_{i}^{n-j-1}) where b_{j} = (j+1)^{1+\alpha} - j^{1-\alpha}
 In the implementation, the summation has explicit history weights:
 \sum_{j=1}^{step-1}(b_{j-1} - b_{j})V_{i,step-j} + b_{step-1}V_{i,0} where b_{j} is as described above
 While fractional PDE (FPDE) models address "long memory" effects in FX markets that standard models miss, 
 they have the following drawbacks: 
 1. High computational complexity: Non-Local Property (fractional derivatives e.g. Caputo derivative solution
 at each time step requires historical data from all previous steps); Memory and Storage (the storage and comp.
 requirements scale significantly as number of time steps increase); Lack of closed form solutions (most FPDE models for FX
 lack analytical solutions, forcing reliance on numerical methods e.g. L1 scheme which would be slow for real-time
 high-frequency FX trading)
 2. Difficult parameter estimation: Alpha (not directly observable and can fluctuate based on market sentiment
 and econ cycles); non-stationarity in FX markets (might need an alpha that changes across time)
 3. Model rigidity and assumptions: Arbitrage risks (certain formulations of fractional Brownian motion used in these models
 are not semi-martingales which can allow for arbitrage opportunities)
 4. Implementation issues: singular kernels (Caputo derivative uses singular kernel that can cause numerical instability); 
 boundary sensitivity (numerical schemes for FPDEs are sensitive to boundary conditions, especially for FX barrier options or
 FX Asian options where non-local memory term complicates early exercise boundary)
*/

mod fractional_pde;

use fractional_pde::{FxOptionParams, solve_fx_tfbs_final_stable};

fn main() {
    // s_max: Max XR in grid, M = no of spatial steps, N = no of time steps (M, N are second, third
    // args in solve_fx_tfbs_final_stable)
    let params = FxOptionParams { s_max: 20.0, k: 1.10, t: 1.0, rd: 0.04, rf: 0.02, sigma: 0.15, alpha: 0.85 };
    let (s, prices) = solve_fx_tfbs_final_stable(params, 400, 200);
    if let Some(pos) = s.iter().position(|&x| x >= 1.10) {
        println!("Stable Price at Spot {:.4}: {:.6}", s[pos], prices[pos]);
    }
}
// We obtain an option price of 0.083560 at spot 1.1141  with the following 
// set of params: s_max: 20.0, k: 1.10, t: 1.0, rd: 0.04, rf: 0.02, sigma: 0.15, alpha: 0.85, M = 400, N = 200
// If the strike price (k) is near the spot, the option is at the money. For an ATM FX option with a 1Y expiration
// and 10%-15% volatility, a premium of 7-8% of the spot is standard. So the calculation in the eg is:
// 1.1141 * 0.075 \approx 0.0835. A premium of 7-8% represents a fair value of an FX option
// with several months to a year of time value remaining (makes sense since in the eg, it's a 1Y option)