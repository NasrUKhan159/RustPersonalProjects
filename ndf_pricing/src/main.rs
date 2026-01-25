// Basic NDF Pricing Example
// Key components: 1. interest rate parity - use domestic and foreign rated to compute fwd price
// 2. Fixing rate: spot rate at maturity, typically determined 2 days prior except for CAD, TRY.
// 3. Cash settlement: no physical delivery - only net pnl difference is exchanged
// 4. Daycount conventions: implementation of ACT/360 or ACT/365 to calculate time to maturity.
fn calculate_ndf_pnl(
    notional: f64,
    contracted_rate: f64,
    fixing_spot_rate: f64,
    settlement_currency_is_base: bool, // e.g., True if USD is base in USD/INR
) -> f64 {
    // Profit/Loss calculation = (Forward Rate - Spot Rate) * Notional
    // NDF settles in cash (USD), no exchange of principal.
    let rate_diff = if settlement_currency_is_base {
        contracted_rate - fixing_spot_rate
    } else {
        (1.0 / contracted_rate) - (1.0 / fixing_spot_rate)
    };
    
    let pnl = rate_diff * notional;
    
    // If PnL > 0, party receives USD. If < 0, pays USD.
    pnl
}

// case 1: standard interest rate parity (IRP) (as shown in calculate_ndf_pnl to compute pnl settlement)
// baseline case where NDF price is derived from spot rate and interest rate differential
// between 2 currencies over a specific time period
fn calculate_irp_ndf_rate(spot: f64, r_domestic: f64, r_foreign: f64, days: f64) -> f64 {
    // Standard IRP Formula: Forward = Spot * (1 + (r_dom * t)) / (1 + (r_for * t))
    let time = days / 360.0;
    spot * (1.0 + r_domestic * time) / (1.0 + r_foreign * time)
}

// case 2: pricing with capital controls and risk premia
// in restricted mkts, NDF rate often deviates from IRP due to country risk/limited capital mobility
// this code adds a basis adjustment/risk premium to the theoretical price to reflect market frictions
fn calculate_adjusted_ndf(irp_rate: f64, country_risk_premium: f64, capital_control_tax: f64) -> f64 {
    // Adjusting for perceived risk and capital barriers
    irp_rate * (1.0 + country_risk_premium + capital_control_tax)
}

// case 3: cash settlement at maturity 
// on the fixing date, PnL is calculated as the difference b/w agreed NDF rate and prevailing spot rate
// and then settled in deliverable currency (usually USD)
fn calculate_settlement_amount(notional_usd: f64, ndf_rate: f64, fixing_rate: f64) -> f64 {
    // Formula: Notional * (Fixing Rate - NDF Rate) / Fixing Rate
    // Result is the amount in USD to be paid or received
    (notional_usd * (fixing_rate - ndf_rate)) / fixing_rate
}

fn main() {
    // basic example of NDF pricing 
    let notional = 1_000_000.0; // USD 1M
    let contracted_rate = 75.0; // USDINR
    let fixing_spot_rate = 76.5; // USDINR
    
    let pnl = calculate_ndf_pnl(notional, contracted_rate, fixing_spot_rate, true);
    
    println!("Net Settlement (USD): {:.2}", pnl); // Negative means payment

    // case 1: standard IRP
    let spot_usd_brl = 5.00; // 1 USD = 5.00 BRL
    let rate_brl = 0.12;      // 12% domestic (BRL)
    let rate_usd = 0.05;      // 5% foreign (USD)
    let days = 90.0;

    let ndf_rate = calculate_irp_ndf_rate(spot_usd_brl, rate_brl, rate_usd, days);
    println!("Standard IRP NDF Rate, Case 1: {:.4}", ndf_rate);

    // case 2: pricing with capital controls and risk premia
    let irp_rate = 5.0841; // Theoretical IRP from Case 1
    let risk_premium = 0.015; // 1.5% extra for political/liquidity risk
    let cap_controls = 0.005; // 0.5% cost from onshore restrictions

    let market_ndf_rate = calculate_adjusted_ndf(irp_rate, risk_premium, cap_controls);
    println!("Adjusted Market NDF Rate, Case 2: {:.4}", market_ndf_rate);

    // case 3: cash settlement at maturity
    let notional = 1_000_000.0; // $1M USD
    let agreed_rate = 5.10;     // NDF Rate from contract
    let fixing_rate = 5.25;     // Actual spot on fixing date

    let settlement = calculate_settlement_amount(notional, agreed_rate, fixing_rate);
    println!("Case 3:");
    if settlement > 0.0 {
        println!("Counterparty pays you: ${:.2} USD", settlement);
    } else {
        println!("You pay counterparty: ${:.2} USD", settlement.abs());
    }
}
