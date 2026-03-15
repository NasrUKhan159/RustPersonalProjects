Personal Academic Project: Avellanada-Stoikov Model for FX Spot pricing

The Avellaneda-Stoikov model is a quantitative framework in high-frequency trading to determine the optimal bid and offer prices for a market maker. 
It focuses on two objectives: maximizing profit from the bid-ask spread and minimizing the risk associated with holding inventory. 

The main things to model in this framework are:
1. Reservation price (r): This is the internal fair value of an asset for the market maker
which is the mid-price adjusted for current inventory. If market-maker has long position, the reservation price is lower than the mid-price
to encourage selling. If they have a short position, it is higher than the mid-price to encourage buying. 
2. Optimal spread (delta): The distance from the reservation price at which the market maker places their bid and ask orders.
3. Inventory Risk Management: The model penalizes large inventory positions by skewing quotes, which helps the trader remain market-neutral.

The modeling framework is the following:
r(s,q,t) = s - q*gamma*sigma^{2}*(T - t)
optimal total spread = delta_a + delta_b = gamma * sigma^{2} * (T - t) + (2/gamma)*ln(1 + (gamma/k))
where g = current mkt mid-price, q = current inventory, sigma = market volatility, gamma = inventory risk aversion parameter, 
T - t = remaining time in trading session, k = order book liquidity/density parameter
(k measures how likely an order is to be filled as it moves away from mid-price)
Applications of Avellanada-Stoikov: Market makers use it to dynamically update limit orders. As trading session reaches its end 
(i.e. as T -> t), spread narrows to ensure market maker can liquidate any remaining inventory before mkt closes.
Key challenge: Calibrating the liquidity parameter k using real order book data

Details of code:
If `inventory` is positive (long), `reservation_price` drops which lowers ask (making one more competitive to sellers since reducing
inventory) and it lowers your bid, making one less attractive to buyers hence avoiding more inventory. Therefore, this handles skew in FX.
Code treats (T - t) as rolling window since market is 24/5


Future extensions to current code:
1. Atomic inventory management: Use std::sync::atomic::AtomicI64 to track inventory (q) across multiple 
execution threads without locking.
2. Pre-calculate the term (2.0 / gamma) * (1.0 + gamma / kappa).ln() as it only changes when liquidity regimes shift.
3. Zero-copy ingest: Use a crate like serde with bincode or faster-hex if you are parsing FIX messages to keep the hot path fast.
4. Add market impact logic to spread calculation for handling larger clip sizes.
5. Gamma can be tuned based on certain conditions. If in some scenario, drawdown limit is tight, gamma should be higher than 0.1.
6. Calibrate k since in FX this measures fill probability. If you move your quote 1 pip away and don't get hit, k is too high.
7. Inventory tracking: Maintain real-time counter of net position 
8. Volatility estimation using rolling window
9. FX specific extensions (out of scope of personal project but real-world considerations):
a. Standard AS assumes a single LOB but FX is OTC so we need a VWAP price from multiple ECNs
b. Dynamic k (since liquidity varies whether in London or NY or Asia, must calibrate k dynamically based on
real-time fill rates and order book density)
c. "Last Look" Adjustment: Many FX venues allow a "last look" period. Modern research suggests adding a penalty term to the 
reservation price or widening the spread to account for the toxicity of flow that exploits this delay.
d. Internalization vs. Externalization: The model should favor internalizing trades (offsetting client buy/sell orders) by 
narrowing spreads when internal flow is balanced, and only widening/skewing when inventory needs to be "hedged" back 
to the interbank market.
e. Using aggregated market data and current q, solve AS equations in high-frequency fashion and stream quotes via FIX
to client GUI or external ECNs.
f. Execution thresholds: Define mininum price buffer to prevent chattering 

References:
1. https://hummingbot.org/blog/guide-to-the-avellaneda--stoikov-strategy/
2. https://www.bayes.citystgeorges.ac.uk/__data/assets/pdf_file/0007/935440/FX-Market-Making.pdf