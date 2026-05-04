Personal Project Tutorial 1: Value an FX instalment
option using stochastic dynamic programming (SDP)
and the Cody algorithm.
- The Cody algorithm is used to compute the 
cumulative normal distribution (CDF) with high
precision - it is not a valuation model for the 
instalment itself.
- Using Cody's algorithm helps us to ensure that 
the tails of the distribution, needed for deep 
OTM or ITM options, are computed w/out underflow/
overflow errors.
- An FX instalment option is a path-dependent derivative
where the premium is paid in discrete parts over time instead
of a lump sum upfront. At each instalment date, holder
decides whether to pay the next instalment to keep the option
alive or let it lapse, making it similar to compound option.
- In addition to vanilla option valuation params, it also takes
instalment schedule as an input: (I, t_n) representing the amount 
and timing of each payment.
- Valuing FX instalment option using stochastic
dynamic programming requires one to model the
problem as a sequence of optimal stopping decisions.
At each instalment date, holder compares cost of
instalment against the continuation value 
- Code: cody_algo.rs
- State space: Grid represents possible XRs S_t.
For high-dimensional problems, the stochastic-rs 
crate for optimised sampling is good.
- Bellman Equation: At each node, holder solves 
V(S,t) = max{0, E[V(S',t+1)] - I} where I is the
instalment amount.
- Numerical stability: Cody's algorithm is preferred
over standard library functions because it avoids
cancelllations in the deep tails of distribution 
(|x| > 5)
- Extensions to tutorial: Handle path-dependent
volatility or jump-diffusion processes. Another way 
to value instalment options is using Laplace transform
because it can be used to solve free boundary problem 
which we see in continuous instalment payments.

Personal Project Tutorial 2: Pricing amortising forward, auto-renewal forward and time option
- Monte Carlo pricer to compute value of amortising FX forward, Monte Carlo pricer to 
compute value of auto-renewal FX forward, a binomial tree pricer to compute value of FX time option
- Amortising forward: Forward where the notional amount decreases over a 
set schedule, summing the discounted risk-neutral payoffs of each amortisation leg.
- Auto-renewal forward: If the spot is in the money at the first expiry, the contract automatically 
renews for a second period.
- Time option (forward window option): Allows holder to execute forward at any point within a specific time window.
This is priced similar to American option but exercise only permitted in the start time to end time window.
- Code: amort_autorenew_timeoption.rs
Extensions to Tutorial: 
- Optimise simulations using parallel processing e.g. Rayon in Rust.

Personal Project Tutorial 3: Retrieve implied volatility from an FX vanilla option
The Black scholes formula for FX vanilla options is:
v(x, K, T, t, \sigma, r_d, r_f, \phi) 
= \phi*exp(-r_d * (T - t))*{f*N(\phi * d_{+} - K*N(\phi * d_{-}))}
where \phi = type, f = x*exp{r_d - r_f}*(T - t) = fwd price of underlying
and d_{+/-} = ((ln(f/K) +/- (\sigma^{2}/2)*(T - t)) / (\sigma * \sqrt{T - t}))
and x = current price of the underlying
The vega of the option (derivative w.r.t. volatility) is:
\partial{v} / \partial{\sigma} = x*exp{-r_f * (T - t)} * \sqrt{T - t} * n(d_{+}) where n() is the pdf of standard gaussian distribution
At maturity, the derivative becomes: x*exp{-r_f * (T)} * \sqrt{T} * n(d_{+}). Extracting the volatility from this using Newton's method
is possible but one needs to be careful about the saddle point with this derivative. This tutorial shows code for 
extracting this volatility using Newton's method, caring about the saddle point.
The key is to select the inflection point volatility (where d_1 * d_2 = 0) as the starting point to ensure Newton's method enables 
algo to start in a region of high sensitivity (max Vega) hence avoiding overshooting in finding final solution.
Code: volretrieve_vanilla.rs
Extensions: Can extend this tutorial to Vanna-Volga pricing to account for the volatility smile.

Personal Project Tutorial 4: More on First and Second Generation Exotics
i. How to price perpetual one-touch, perpetual no-touch, down-and-out knock-out put option. How do changes in the spot price (case of 
spot price going up or down), and/or the forward price becoming different from the spot (or staying the same), affect the value of a down-and-out knock-out put option?
Perpetual One-Touch & No-Touch: In a perpetual setting (no expiry), these options are priced based on the probability of the spot 
S reaching barrier H. 
One-touch: Pays 1 if S hits H - if S moving towards H, value is (S/H)^{\alpha} where \alpha depends on the drift and volatility.
No-touch: Pays 1 if H is never hit. In perpetual model with drift towards barrier, this is typically 0 but if drift is away, it is 1 - (S/H)^{\alpha}
Down-and-Out Knock-Out Put (DOP): Standard put that expires worthless if spot hits a lower barrier H.
Sensitivity analysis of DOP: If spot rises, option value generally falls. While risk of "knocking out" at lower barrier decreases, the put becomes more OTM.
If spot falls, value increases initially as put becomes more ITM but as spot approaches barrier H, value drops towards zero creating negative gamma near barrier.
Impact of forward all-in vs spot price: F = S * exp{r_d - r_f} * T
All-in > spot => r_d > r_f so market expects spot to drift up, making lower barrier less likely to hit (i.e. reducing prob of knock-out), increasing value of DOP compared to F = S.
All-in < spot => r_d < r_f so mkt expects spot to drift down, increasing prob of knock-out, causing DOP's value to be much lower than standard vanilla put.
All-in = spot => Option val driven by volatility and distance to barrier without impact of interest rate differentials.
If the EURGBP rates are negatively correlated with the GBP rates, how does this change the TV of a 
strike-out put (also called a knock-out put)? For EURGBP DOP, negative corr b/w XR and interest rate (r_GBP) generally decreases TV from 3 different angles:
NB: For some currency pair, the base currency refers to the foreign currency and the term currency refers to the domestic currency.
Angle 1: Increased knock-out probability: As EURGBP drops towards lower barrier, corrresponding rise in GBP rates increases forward drift away from the barrier. However, the negative 
correlation makes hitting barrier during high-volatility spikes more likely.
Angle 2: If EURGBP XR falls, and consequently if GBP interest rates rise, a higher GBP interest rate relative to EUR interest rate results in forward premium (forward > spot). If forward price goes higher
than spot, option's expected intrinsic value falls.  
Angle 3: If r_d rises when EURGBP falling, put option going ITM, the payoff of the put is discounted back to PV using r_d. So increased ITM value is 
offset by higher discount factor, reducing the TV.
Extensions: 
- Explore with different params and different methods other than Cholesky to link FX moves to interest rate moves.
- Adjust to include stochastic volatility for more accurate barrier pricing.
Code: first_gen_exotic_px.rs 
ii. Value a forward-start option where strike = spot rate + d for some real number d, and compare values with the standard 
value of a forward start option. Back out forward volatility smile using forward start option values. 
Value window barrier options where strike and barrier are multiples of the spot rate S_{t}.
Do the same for standard barrier options and compare results b/w both types of options
For a forward-start option with strike K = S_{t_{1}} + d (where t_{1} is the future start date and T is the maturity)
the payoff at T is max{S_{T} - (S_{t_{1}} + d,), 0}:
- Standard forward-start: Strike K = k * S_{t_{1}} where k = 1 => ATM. This is homogeneous of degree 1.
- Strike S_{t_{1}} + d: This is not homogeneous of degree 1 in S_{t_{1}} unless d = 0. The absolute difference d makes the option value dependent on the expected
level of S_{t_{1}}.
- Window vs standard barrier options: In standard barrier options, the barrier is monitored continuously throughout the life of option. In a window barrier option, the barrier is only
active during a specific "window" of time within option's life. Window barriers are generally more expensive for knock-out events than standard barriers because there is less time for the 
knock event to occur, resulting in higher premiums for knock-out structures.
- Rust code uses MC simulation to value forward start and barrier options. For forward start option with strike S_{t_{1}} + d, value deviates from the standard proportionality because
d breaks scaling symmetry of Black Scholes. It also implements Newton's method for deriving forward volatility smile.
Extensions to this:
- Choice of strike (k) and market price to back out the implied volatility. If we change these values to say for e.g.
smaller market price for each strike value e.g. P = 25.41 for k = 0.8 instead of P = 48.50 - what impact does this have on the convergence and why?
Code: first_gen_exotic_px_2.rs 
iii. Suppose exchange rate follows Brownian motion without drift and constant volatility. Implement hedging methods for 
single one-touch with digitals. Do the same with the following other cases: a. Brownian motion with drift and/or volatility
b. Double one-touch with digitals and compare results.
Used static replication for implementing hedging for one-touch digital options.
- We cannot do dynamic delta hedging for digitals because of infinite delta at the barrier
- Static replication involves creating a portfolio of vanilla options (or other digitals) that match payoff of barrier option at boundary
One-Touch (Digital): This pays a fixed amount if the exchange rate S_{t} touches a barrier B before expiry T
Hedging: It is equivalent to a digital call (for up-and-in) with a strike at the barrier, but since it's "one-touch" (American-style), 
it is often hedged using a portfolio of European digital options or a "risk-reversal" style setup that mimics the first-passage time probability.
Double One-Touch (DOT): This pays if S_{t} touches either an upper barrier U or a lower barrier L
Hedging: This is typically replicated using a series of digital calls and puts. Because the barriers can be hit at any time, the replication 
requires a strip of options across different strikes and maturities to capture the "touch" probability at any point in the path.
Code: first_gen_exotic_px_3.rs
Second generation exotics: 
a. Value spread options in the following different cases: 
case 1: joint base currency, both currency pairs are directs (USDHKD, USDSGD)
case 2: joint currency in both pairs, base ccy in pair 1 = term ccy in pair 2 (USDNOK, GBPUSD)
case 3: joint currency in both pairs, term ccy in pair 1 = base ccy in pair 2 (AUDUSD, USDSEK)
case 4: no joint currency, both directs (NZDUSD, EURPLN)
case 5: no joint currency, both crosses (AUDNZD, CHFJPY)
Used Kirk's approximation for valuing spread options (assuming assets are log-normal)
Extensions: 
- Integrate correlation matrix to handle volatility of derived cross rates more accurately.
- Input a stream of prices instead of static prices
Code: second_gen_exotic_px.rs 
b. Value Bermudan style corridor using Garman-Kohlhagen, and find market price based on theoretical fair value.
Code: second_gen_exotic_px_2.rs
Note: We get theta = 0 in the example, because in a binomial tree, if the time bump (dt) is small enough that the "spot"
does not cross any new node boundaries between T and T - dt, the tree structure remains identical. For a binary payoff, the option value 
will be the same, resulting in zero theta. We get a very large negative value for gamma (-759.53) because we are dealing with a digital/binary
payoff. In a corridor (1.05 to 1.15) with the spot at 1.1, we are "short" gamma at the boundaries. If the bump `ds` is too small, only end up
catching one node jump, making Gamma look like massive spike or flat line.
Extensions:
- Experiment with different inputs to EURUSD FX bermudan option and see what impact this has on the Greeks values
- Experiment with a variety of r, t, rho, k values.
- Include vega in the Greeks calculations 
- Experiment with a stream of parameter values instead of static values and note how the results change!
- Increase `steps` to more (e.g. greater than 500) because using a binomial tree to price a corridor can cause "digital" (sharp) edges. 
So this will give smoother Greeks values. 
- Calibrate the volatility surface using market data (vol surface interpolation to find mkt implied vol)

Personal Project Tutorial 5: More complicated issues in FX pricing
compute the vanna of a butterfly and the volga or a risk reversal and examine under what conditions they are close to zero and 
conversely, which parameter scenarios would cause problems in the pricing of vanilla options. Along these lines, discuss the 
alternative of hedging vega, vanna and volga simply with three options, one at-the-money, one 25-delta call and one 25-delta put.
In the FX market, the smile is quoted using Risk Reversal (RR) and Butterfly (BF). Strikes are not fixed but are derived from
delta (e.g. 25-delta call/put) which depends on model and current volatility.
- When Greeks are zero, vanna is near zero for ATM forward options but only if skew is symmetric. 
- Volga is minimized near the ATM strike - if the butterfly has zero volga, the market is pricing no "convexity" or tail risk beyond Black-Scholes.
- We face pricing problems in the case of high interest rate differentials that push the forward price far from the spot.
- For exotic FX products e.g. no-touch and knock-out options, vanna-volga adjustments can become unstable near the barrier. 
- Hedging with an ATM, a 25-delta call and a 25-delta put is the standard way to replicate the market's smile cost.
- This creates the following system of 3 linear equations:
w_{1}*Vega_{ATM} + w_{2}*Vega_{25C} + w_{3}*Vega_{25P} = Vega_{target} (1)
w_{1}*Vanna{ATM} + w_{2}*Vanna_{25C} + w_{3}*Vanna_{25P} = Vanna_{target} (2)
w_{1}*Volga_{ATM} + w_{2}*Volga_{25C} + w_{3}*Volga_{25P} = Volga_{target} (3)
- In risk reversal, 25-delta call - 25-delta put isolates vanna (skew), and the average of 25-delta call and 25-delta put, minus
ATM isolates volga (curvature).
- Code solves for the hedge weights for example combinations of vega, vanna and volga.
Extensions:
- Experiment with different combinations of vega, vanna and volga 
- Extend to automatically determine 25-delta strikes based on current volatility and interest rate differentials.
Code: more_issues_fx_pricing_1.rs

References:
1. FX Options and Structured Products by Uwe Wystup
2. "Binomial Method in Bermudan Option" by E. Siswanah, A.M. Idrus, M. Malik Hakim
3. "Hedging at-the-money digital options near maturity" by A Blanc-Bocquel (2023)
4. "Robust hedging of double touch barrier options" by A.M.G. Cox and J.K. Obłój
5. https://ideas.repec.org/a/eee/ejores/v201y2010i1p222-230.html (European Journal of Operational Research)
6. https://arxiv.org/pdf/1710.11232 (ArXiv: Option Pricing research)
7. https://crates.io/crates/optionstratlib (Rust Library)
8. https://quant.stackexchange.com/questions/39715/how-to-simulate-fx-forwards
9. https://quant.stackexchange.com/questions/17083/what-is-the-fair-price-of-this-option
10. https://quant.stackexchange.com/questions/18565/how-can-one-value-a-bermudan-option
11. https://www.investopedia.com/terms/d/daoo.asp
12. https://www.investopedia.com/terms/b/barrieroption.asp
13. https://www.chathamfinancial.com/insights/fx-option-pricing
14. https://www.simtrade.fr/blog_simtrade/pricing-barrier-options-simulations-sensitivity-analysis-greeks/
15. https://milltech.com/resources/glossary/forward-points
16. https://www.quantstart.com/articles/Implied-Volatility-in-C-using-Template-Functions-and-Newton-Raphson/
17. https://www.rdocumentation.org/packages/RTL/versions/1.3.7/topics/spreadOption
18. https://www.quantpie.co.uk/fx/fx_rr_str.php
19. https://quantpie.co.uk/fx/fx_smile_vv.php
20. https://www.youtube.com/watch?v=--mnmjNMeBA